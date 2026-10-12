// 合成自检：用模拟器真实物理模型生成带噪相位，比较各 local_model 的预测误差与就绪时间。
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
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> ua(0.780,1.045), uw(1.884,2.000), uphi(0,2*PI), uth(0,2*PI);
  std::normal_distribution<double> noise(0.0, 2.0*PI/180.0); // 2 deg
  std::uniform_real_distribution<double> uni(0,1);
  const double dt=1.0/30.0;
  int modes[3]={7,0,4};
  const char* names[3]={"phys(7)","csine(0)","AR(4)"};
  for(int mi=0; mi<3; ++mi){
    std::vector<double> errs; int ready_cnt=0, tot=0;
    // 就绪时间统计
    std::vector<double> ready_times;
    for(int trial=0; trial<400; ++trial){
      const double a=ua(rng), w=uw(rng), phi=uphi(rng), th0=uth(rng);
      const double Tfit=1.2; int M=(int)std::lround(Tfit/dt);
      LocalRuneTracker tr; tr.set_mode(modes[mi]); tr.set_min_span_deg(20.0);
      double t_ready=-1;
      for(int k=0;k<M+10;++k){
        double t=k*dt;
        // 加一点相位随机游走模拟 PnP 抖动
        double truth=th0 + ((2.090-a)*t + (a/w)*(std::cos(phi)-std::cos(w*t+phi)));
        double z=wrap(truth+noise(rng));
        if(uni(rng)<0.05) z=wrap(z + (uni(rng)<0.5?-1:1)*1.0); // 5% 1rad 野值
        if(k<M) tr.push(t,z);
        if(t_ready<0 && tr.ready()) t_ready=t;
      }
      double tf=(M-1)*dt+0.30;
      double truth=th0 + ((2.090-a)*tf + (a/w)*(std::cos(phi)-std::cos(w*tf+phi)));
      double pred;
      ++tot;
      if(tr.predict(tf,&pred)){
        ++ready_cnt;
        errs.push_back(std::abs(wrap(pred-truth))*180.0/PI);
      }
      if(t_ready>0) ready_times.push_back(t_ready);
    }
    std::sort(errs.begin(),errs.end());
    std::sort(ready_times.begin(),ready_times.end());
    double med = errs.empty()?-1:errs[errs.size()/2];
    double p90 = errs.empty()?-1:errs[std::min((int)errs.size()-1,(int)(0.9*errs.size()))];
    double rt = ready_times.empty()?-1:ready_times[ready_times.size()/2];
    std::printf("%-10s ready=%3d/%d  预测+0.30s: 中位 %6.2f deg  p90 %6.2f deg  (就绪中位 %.2f s)\n",
                names[mi], ready_cnt, tot, med, p90, rt);
  }
  return 0;
}
