// 合成自检：模拟器真值 + 噪声 + 【持续换支/翻解】异常，比较模型锚定创新门 on/off。
// 编译: g++ -O2 -std=c++17 -I src -I /usr/include/eigen3 tools/phys_selftest.cpp src/local_tracker.cpp -o /tmp/phys_selftest
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include "local_tracker.hpp"

static const double PI = 3.14159265358979323846;
static double wrap(double a){ while(a>PI)a-=2*PI; while(a<=-PI)a+=2*PI; return a; }

int main(){
  std::mt19937 rng(2026);
  std::uniform_real_distribution<double> ua(0.780,1.045), uw(1.884,2.000), uphi(0,2*PI), uth(0,2*PI);
  std::normal_distribution<double> noise(0.0, 0.035);  // ~2 deg
  std::uniform_real_distribution<double> uni(0,1);
  const double dt=1.0/30.0;
  const double Tfit=1.2; const int M=(int)std::lround(Tfit/dt);

  for (double gate_deg : {1e9, 29.0}) {
    std::vector<double> errs; int ready=0; int total=0;
    for (int trial=0; trial<500; ++trial) {
      const double a=ua(rng), w=uw(rng), phi=uphi(rng), th0=uth(rng);
      LocalRuneTracker tr; tr.set_mode(7); tr.set_min_span_deg(20.0);
      if (gate_deg < 1e8) tr.set_phys_gate_rad(gate_deg*PI/180.0);
      // 持续换支：30% 概率在 t0 处加一个偏移，持续 L 帧
      const bool shift = uni(rng) < 0.30;
      const double shift_t0 = 0.45 + 0.5*uni(rng);
      const int shift_len = 5 + (int)(uni(rng)*25);
      const double shift_off = (uni(rng)<0.5?-1:1) * (2.0 + 1.5*uni(rng));  // 115~200 deg
      for (int k=0;k<M+10;++k) {
        const double t=k*dt;
        double truth=th0 + ((2.090-a)*t + (a/w)*(std::cos(phi)-std::cos(w*t+phi)));
        double z=wrap(truth+noise(rng));
        if (uni(rng)<0.02) z=wrap(z + (uni(rng)<0.5?-1:1)*1.0);   // 2% 单帧尖峰
        if (shift && t>=shift_t0 && t<shift_t0+shift_len*dt) z=wrap(z+shift_off); // 持续换支
        if (k<M) tr.push(t,z);
      }
      const double tf=(M-1)*dt+0.30;
      const double truth=th0 + ((2.090-a)*tf + (a/w)*(std::cos(phi)-std::cos(w*tf+phi)));
      double pred; ++total;
      if (tr.predict(tf,&pred)) { ++ready; errs.push_back(std::abs(wrap(pred-truth))*180.0/PI); }
    }
    std::sort(errs.begin(),errs.end());
    double med = errs.empty()?-1:errs[errs.size()/2];
    double p90 = errs.empty()?-1:errs[std::min((int)errs.size()-1,(int)(0.9*errs.size()))];
    std::printf("gate=%6.0f deg: ready=%3d/%d  预测+0.30s 中位 %6.2f deg  p90 %6.2f deg\n",
                gate_deg<1e8?gate_deg:0.0, ready, total, med, p90);
  }
  return 0;
}
