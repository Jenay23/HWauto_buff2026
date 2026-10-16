#include "reborn_io.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace reborn {
namespace {

constexpr size_t kHandshakeLen = 32;
constexpr size_t kNoticeLen = 16;
constexpr size_t kFrameHeaderLen = 96;
constexpr size_t kTelemetryLen = 43;
constexpr size_t kCommandLen = 29;

int64_t now_unix_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int64_t now_mono_ns() {
  timespec ts{};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

template <typename T>
T load_le(const uint8_t* p) {
  T v{};
  std::memcpy(&v, p, sizeof(T));
  return v;
}

void store_le(uint8_t* p, const void* v, size_t n) { std::memcpy(p, v, n); }

}  // namespace

uint16_t crc16(const uint8_t* data, size_t n) {
  uint16_t crc = 0xffff;
  for (size_t i = 0; i < n; ++i) {
    crc ^= static_cast<uint16_t>(data[i]);
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 1) ? static_cast<uint16_t>((crc >> 1) ^ 0x8408) : static_cast<uint16_t>(crc >> 1);
    }
  }
  return crc;
}

// ---------------------------------------------------------------- Camera ----

void Camera::close() {
  if (map_ != nullptr && map_ != MAP_FAILED) munmap(const_cast<uint8_t*>(map_), map_len_);
  map_ = nullptr;
  map_len_ = 0;
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
  slots_ = 0;
  slot_size_ = 0;
  session_ = 0;
  have_sequence_ = false;
}

bool Camera::read_exact(void* dst, size_t n, int timeout_ms) const {
  uint8_t* out = static_cast<uint8_t*>(dst);
  size_t done = 0;
  const int64_t deadline = now_mono_ns() + static_cast<int64_t>(timeout_ms) * 1000000LL;
  while (done < n) {
    int64_t left_ms = (deadline - now_mono_ns()) / 1000000LL;
    if (left_ms < 0) left_ms = 0;
    pollfd pfd{};
    pfd.fd = fd_;
    pfd.events = POLLIN;
    const int pr = ::poll(&pfd, 1, static_cast<int>(left_ms));
    if (pr == 0) return false;
    if (pr < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    const ssize_t r = ::read(fd_, out + done, n - done);
    if (r > 0) {
      done += static_cast<size_t>(r);
      continue;
    }
    if (r == 0) return false;  // peer closed
    if (errno == EINTR || errno == EAGAIN) continue;
    return false;
  }
  return true;
}

bool Camera::open(const Config& cfg, std::string* err) {
  close();
  const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    *err = "socket() failed";
    return false;
  }
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (cfg.socket.size() >= sizeof(addr.sun_path)) {
    *err = "camera socket path too long";
    ::close(fd);
    return false;
  }
  std::memcpy(addr.sun_path, cfg.socket.c_str(), cfg.socket.size() + 1);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    *err = "connect(" + cfg.socket + ") failed: " + std::strerror(errno) +
           " (is the simulator running?)";
    ::close(fd);
    return false;
  }
  fd_ = fd;

  uint8_t hs[kHandshakeLen];
  if (!read_exact(hs, sizeof(hs), 5000)) {
    *err = "no RBIP handshake from the simulator";
    close();
    return false;
  }
  if (std::memcmp(hs, "RBIP", 4) != 0) {
    *err = "bad RBIP magic (simulator too old? RBIP v1 is rejected)";
    close();
    return false;
  }
  const uint16_t version = load_le<uint16_t>(hs + 4);
  const uint16_t header_len = load_le<uint16_t>(hs + 6);
  slots_ = load_le<uint32_t>(hs + 8);
  slot_size_ = load_le<uint64_t>(hs + 12);
  const uint32_t path_len = load_le<uint32_t>(hs + 20);
  const uint64_t reserved = load_le<uint64_t>(hs + 24);
  if (version != 2 || header_len != kHandshakeLen || reserved != 0 || slots_ < 2 ||
      slot_size_ == 0 || slot_size_ % 4096 != 0 || path_len == 0 || path_len > 4096) {
    *err = "unsupported RBIP handshake (need v2)";
    close();
    return false;
  }

  std::vector<char> path(path_len + 1, '\0');
  if (!read_exact(path.data(), path_len, 5000)) {
    *err = "truncated RBIP shm path";
    close();
    return false;
  }
  const std::string shm_path(path.data());
  if (shm_path != cfg.shm) {
    std::fprintf(stderr, "[reborn] simulator shm path is %s (configured %s)\n", shm_path.c_str(),
                 cfg.shm.c_str());
  }

  struct stat st{};
  if (::stat(shm_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
    *err = "shm " + shm_path + " is not a regular file";
    close();
    return false;
  }
  const uint64_t need = slot_size_ * static_cast<uint64_t>(slots_);
  if (static_cast<uint64_t>(st.st_size) < need) {
    *err = "shm " + shm_path + " is smaller than the advertised ring";
    close();
    return false;
  }
  const int shm_fd = ::open(shm_path.c_str(), O_RDONLY);
  if (shm_fd < 0) {
    *err = "open(" + shm_path + ") failed: " + std::strerror(errno);
    close();
    return false;
  }
  map_len_ = static_cast<size_t>(need);
  void* m = ::mmap(nullptr, map_len_, PROT_READ, MAP_SHARED, shm_fd, 0);
  ::close(shm_fd);
  if (m == MAP_FAILED) {
    *err = "mmap(" + shm_path + ") failed";
    map_len_ = 0;
    close();
    return false;
  }
  map_ = static_cast<const uint8_t*>(m);
  return true;
}

bool Camera::release(uint32_t slot, uint64_t sequence) const {
  uint8_t ack[kNoticeLen];
  std::memcpy(ack, "RBRA", 4);
  store_le(ack + 4, &slot, 4);
  store_le(ack + 8, &sequence, 8);
  size_t done = 0;
  while (done < sizeof(ack)) {
    const ssize_t w = ::write(fd_, ack + done, sizeof(ack) - done);
    if (w > 0) {
      done += static_cast<size_t>(w);
      continue;
    }
    if (w < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    return false;
  }
  return true;
}

bool Camera::recv(Frame* out, int timeout_ms) {
  if (fd_ < 0 || map_ == nullptr) return false;

  uint8_t notice[kNoticeLen];
  if (!read_exact(notice, sizeof(notice), timeout_ms)) return false;
  if (std::memcmp(notice, "RBFR", 4) != 0) {
    // A damaged notice has no slot to release; dropping the connection is the
    // documented behaviour.
    std::fprintf(stderr, "[reborn] damaged RBFR notice, reconnecting\n");
    close();
    return false;
  }
  const uint32_t slot = load_le<uint32_t>(notice + 4);
  const uint64_t sequence = load_le<uint64_t>(notice + 8);
  if (slot >= slots_) {
    std::fprintf(stderr, "[reborn] notice slot %u out of range\n", slot);
    close();
    return false;
  }

  const uint8_t* base = map_ + static_cast<size_t>(slot) * slot_size_;
  const uint8_t* hdr = base;
  const bool magic_ok = std::memcmp(hdr, "RBCAM\0\0\0", 8) == 0;
  const uint16_t frame_version = load_le<uint16_t>(hdr + 8);
  const uint16_t frame_header_len = load_le<uint16_t>(hdr + 10);
  const uint32_t pixel_format = load_le<uint32_t>(hdr + 12);
  const uint32_t width = load_le<uint32_t>(hdr + 16);
  const uint32_t height = load_le<uint32_t>(hdr + 20);
  const uint32_t stride = load_le<uint32_t>(hdr + 24);
  const uint32_t pixel_bytes = load_le<uint32_t>(hdr + 28);
  const uint64_t header_sequence = load_le<uint64_t>(hdr + 40);
  const uint64_t source_mono = load_le<uint64_t>(hdr + 48);

  const bool frame_ok = magic_ok && frame_version == 1 && frame_header_len == kFrameHeaderLen &&
                        pixel_format == 1 && width > 0 && height > 0 && stride >= width * 3 &&
                        static_cast<uint64_t>(pixel_bytes) + kFrameHeaderLen <= slot_size_ &&
                        header_sequence == sequence;

  if (!frame_ok) {
    release(slot, sequence);  // bad frames must be acknowledged too
    static int reported = 0;
    if (reported++ < 5) {
      std::fprintf(stderr,
                   "[reborn] bad frame: magic=%d ver=%u hlen=%u fmt=%u %ux%u stride=%u bytes=%u "
                   "seq=%llu/%llu\n",
                   magic_ok ? 1 : 0, frame_version, frame_header_len, pixel_format, width, height,
                   stride, pixel_bytes, static_cast<unsigned long long>(header_sequence),
                   static_cast<unsigned long long>(sequence));
    }
    return false;
  }
  if (have_sequence_ && sequence <= last_sequence_) {
    release(slot, sequence);
    return false;
  }

  out->bimg.resize(20 + static_cast<size_t>(width) * height * 3);
  uint8_t* rgb = out->bimg.data() + 20;
  for (uint32_t y = 0; y < height; ++y) {
    const uint8_t* src = base + kFrameHeaderLen + static_cast<size_t>(y) * stride;
    uint8_t* dst = rgb + static_cast<size_t>(y) * width * 3;
    for (uint32_t x = 0; x < width; ++x) {  // BGR8 -> RGB888
      dst[x * 3 + 0] = src[x * 3 + 2];
      dst[x * 3 + 1] = src[x * 3 + 1];
      dst[x * 3 + 2] = src[x * 3 + 0];
    }
  }

  // The source clock is CLOCK_MONOTONIC; the bridge's staleness gate uses the
  // system clock, so rebuild the capture instant in unix time using the
  // measured in-flight delay. Never re-anchor per frame beyond that.
  const int64_t mono_now = now_mono_ns();
  int64_t latency = (mono_now > 0 && source_mono > 0) ? mono_now - static_cast<int64_t>(source_mono) : 0;
  if (latency < 0) latency = 0;
  out->unix_ns = now_unix_ns() - latency;
  out->width = width;
  out->height = height;
  out->sequence = sequence;
  out->fx = 0.0;
  out->fy = 0.0;
  out->cx = 0.0;
  out->cy = 0.0;
  double f[4];
  for (int i = 0; i < 4; ++i) std::memcpy(&f[i], hdr + 64 + i * 8, 8);
  out->fx = f[0];
  out->fy = f[1];
  out->cx = f[2];
  out->cy = f[3];
  session_ = load_le<uint64_t>(hdr + 32);

  std::memcpy(out->bimg.data(), "BIMG", 4);
  store_le(out->bimg.data() + 4, &out->unix_ns, 8);
  store_le(out->bimg.data() + 12, &width, 4);
  store_le(out->bimg.data() + 16, &height, 4);

  last_sequence_ = sequence;
  have_sequence_ = true;
  release(slot, sequence);
  return true;
}

// ---------------------------------------------------------------- Serial ----

bool Serial::open(const std::string& device, std::string* err) {
  close();
  const int fd = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd < 0) {
    *err = "open(" + device + ") failed: " + std::strerror(errno) +
           " (run scripts/setup-virtual-gimbal.py and start the simulator first)";
    return false;
  }
  termios tio{};
  if (tcgetattr(fd, &tio) == 0) {
    cfmakeraw(&tio);  // the wire carries binary frames; no CR/NL or XON/XOFF
    tio.c_cflag |= CLOCAL | CREAD;
    tio.c_cc[VMIN] = 0;
    tio.c_cc[VTIME] = 0;
    cfsetispeed(&tio, B115200);
    cfsetospeed(&tio, B115200);
    tcsetattr(fd, TCSANOW, &tio);
    tcflush(fd, TCIOFLUSH);
  }
  fd_ = fd;
  stop_.store(false);
  reader_ = std::thread([this] { reader_loop(); });
  return true;
}

void Serial::close() {
  stop_.store(true);
  if (reader_.joinable()) reader_.join();
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
  std::lock_guard<std::mutex> lk(mu_);
  have_ = false;
  count_ = 0;
}

void Serial::reader_loop() {
  std::vector<uint8_t> buf;
  buf.reserve(256);
  uint8_t chunk[256];
  while (!stop_.load()) {
    pollfd pfd{};
    pfd.fd = fd_;
    pfd.events = POLLIN;
    const int pr = ::poll(&pfd, 1, 50);
    if (pr <= 0) continue;
    const ssize_t r = ::read(fd_, chunk, sizeof(chunk));
    if (r <= 0) continue;
    buf.insert(buf.end(), chunk, chunk + r);

    // Bounded streaming parse: invalid frames advance one byte so an "SP" after
    // noise is still found.
    size_t i = 0;
    while (buf.size() - i >= kTelemetryLen) {
      if (buf[i] != 'S' || buf[i + 1] != 'P') {
        ++i;
        continue;
      }
      const uint8_t* p = buf.data() + i;
      const uint16_t want = load_le<uint16_t>(p + kTelemetryLen - 2);
      if (crc16(p, kTelemetryLen - 2) != want) {
        ++i;
        continue;
      }
      float q[4];
      for (int k = 0; k < 4; ++k) std::memcpy(&q[k], p + 3 + k * 4, 4);
      float yaw, yaw_vel, pitch, pitch_vel, bullet_speed;
      std::memcpy(&yaw, p + 19, 4);
      std::memcpy(&yaw_vel, p + 23, 4);
      std::memcpy(&pitch, p + 27, 4);
      std::memcpy(&pitch_vel, p + 31, 4);
      std::memcpy(&bullet_speed, p + 35, 4);
      (void)yaw_vel;
      (void)pitch_vel;
      (void)bullet_speed;
      const float w = q[0], x = q[1], y = q[2], z = q[3];
      const float roll = std::atan2(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y));
      {
        std::lock_guard<std::mutex> lk(mu_);
        yaw_ = yaw;      // muzzle frame: yaw about up, left positive
        pitch_ = pitch;  // muzzle frame: pitch is DOWN positive, as the bridge wants
        roll_ = roll;
        ++count_;
        have_ = true;
      }
      i += kTelemetryLen;
    }
    if (i > 0) buf.erase(buf.begin(), buf.begin() + static_cast<long>(i));
    if (buf.size() > 4096) buf.erase(buf.begin(), buf.end() - kTelemetryLen);
  }
}

bool Serial::latest(float* yaw, float* pitch, float* roll, uint64_t* count) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_) return false;
  if (yaw) *yaw = yaw_;
  if (pitch) *pitch = pitch_;
  if (roll) *roll = roll_;
  if (count) *count = count_;
  return true;
}

bool Serial::send(double yaw, double pitch, bool fire, bool control) {
  if (fd_ < 0) return false;
  uint8_t pkt[kCommandLen];
  std::memcpy(pkt, "SP", 2);
  const uint8_t mode = !control ? 0 : (fire ? 2 : 1);
  pkt[2] = mode;
  const float yaw_f = static_cast<float>(yaw);
  const float pitch_f = static_cast<float>(pitch);
  const float zero = 0.0f;
  store_le(pkt + 3, &yaw_f, 4);
  store_le(pkt + 7, &zero, 4);
  store_le(pkt + 11, &zero, 4);
  store_le(pkt + 15, &pitch_f, 4);
  store_le(pkt + 19, &zero, 4);
  store_le(pkt + 23, &zero, 4);
  const uint16_t crc = crc16(pkt, kCommandLen - 2);
  store_le(pkt + kCommandLen - 2, &crc, 2);

  size_t done = 0;
  while (done < sizeof(pkt)) {
    const ssize_t w = ::write(fd_, pkt + done, sizeof(pkt) - done);
    if (w > 0) {
      done += static_cast<size_t>(w);
      continue;
    }
    if (w < 0 && errno == EINTR) continue;
    if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      pollfd pfd{};
      pfd.fd = fd_;
      pfd.events = POLLOUT;
      if (::poll(&pfd, 1, 5) > 0) continue;
    }
    return false;
  }
  return true;
}

}  // namespace reborn
