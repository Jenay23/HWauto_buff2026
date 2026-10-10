// 最小验证：Zenoh C 1.9 API 用法（订阅 rt/gimbal_pose + 发布 cmd/gimbal）
#include <zenoh.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <string>

static std::mutex g_mu;
static std::string g_last_pose;
static std::atomic<int> g_pose_count{0};

static void on_pose(struct z_loaned_sample_t *sample, void *ctx) {
  (void)ctx;
  const struct z_loaned_bytes_t *pl = z_sample_payload(sample);
  struct z_loaned_slice_t *slice = nullptr;
  struct z_owned_slice_t owned;
  if (z_bytes_to_slice(pl, &owned) != Z_OK) return;
  const uint8_t *d = z_slice_data(z_loan(owned));
  size_t n = z_slice_len(z_loan(owned));
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_last_pose.assign(reinterpret_cast<const char*>(d), n);
  }
  g_pose_count++;
  z_drop(z_move(owned));
  (void)slice;
}

int main() {
  z_owned_config_t cfg;
  z_config_default(&cfg);
  z_owned_session_t session;
  if (z_open(&session, z_move(cfg), NULL) != Z_OK) { printf("[probe] z_open FAILED\n"); return 1; }
  printf("[probe] session open\n");

  z_owned_closure_sample_t cb;
  z_closure_sample(&cb, on_pose, NULL, NULL);
  z_view_keyexpr_t ke;
  z_view_keyexpr_from_str(&ke, "rt/gimbal_pose");
  z_owned_subscriber_t sub;
  if (z_declare_subscriber(z_loan(session), &sub, z_loan(ke), z_move(cb), NULL) != Z_OK) {
    printf("[probe] declare_subscriber FAILED\n"); return 2;
  }
  printf("[probe] subscribed rt/gimbal_pose, waiting 3 s ...\n");

  // 发布器
  z_owned_publisher_t pub;
  z_view_keyexpr_t ke2; z_view_keyexpr_from_str(&ke2, "cmd/gimbal");
  bool have_pub = z_declare_publisher(z_loan(session), &pub, z_loan(ke2), NULL) == Z_OK;
  printf("[probe] publisher cmd/gimbal: %s\n", have_pub ? "ok" : "FAILED");

  for (int i = 0; i < 30; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (have_pub && i == 10) {
      const char *j = "{\"yaw\":0.0,\"pitch\":-0.05,\"roll\":0.0,\"fire\":false,\"control\":false}";
      z_owned_bytes_t b; z_bytes_copy_from_buf(&b, (const uint8_t*)j, std::strlen(j));
      z_put(z_loan(session), z_loan(ke2), z_move(b), NULL);
      printf("[probe] sent control=false probe\n");
    }
  }
  {
    std::lock_guard<std::mutex> lk(g_mu);
    printf("[probe] RESULT: received %d gimbal_pose msgs; last=\"%s\"\n",
           g_pose_count.load(), g_last_pose.c_str());
  }
  printf("[probe] %s\n", g_pose_count.load() > 0 ? ">>> API OK" : ">>> NO DATA (simulator not running?)");
  return 0;
}
