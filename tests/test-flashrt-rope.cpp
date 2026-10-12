// SPDX-License-Identifier: MIT
// Tests for the bounded FlashRT FP16-math / GGML F32 NEOX RoPE adapter.
// The host-only build has no ggml library dependency: it tests POD metadata.
#include "ggml.h"
#include "ggml-cuda/flashrt-rope-guard.h"
#ifndef FLASHRT_ROPE_HOST_ONLY
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
static void need(bool ok, const std::string & text) { if (!ok) { throw std::runtime_error(text); } }
struct options { bool host_only=false, graphs=false; int mode=-1; std::string prefix; };
static options parse(int argc, char ** argv) {
    options o;
    for (int i=1;i<argc;++i) {
        std::string k=argv[i];
        if (k=="--host-only") { o.host_only=true; continue; }
        if (k=="--help") { std::cout<<"--host-only OR --mode 0|1 --graphs off|on --output-prefix PATH\n"; std::exit(0); }
        need(i+1<argc,"missing value for "+k); std::string v=argv[++i];
        if (k=="--mode") { need(v=="0"||v=="1","mode must be 0 or 1"); o.mode=v=="1"; }
        else if(k=="--graphs") { need(v=="off"||v=="on","graphs must be off or on"); o.graphs=v=="on"; }
        else if(k=="--output-prefix") { o.prefix=v; }
        else { throw std::runtime_error("unknown option "+k); }
    }
    need(o.host_only||(o.mode>=0&&!o.prefix.empty()),"mode and output-prefix required"); return o;
}
struct case_spec {
    std::string name="rope_h1_m11"; int head_dim=256, heads=1, rows=11, rope_mode=GGML_ROPE_TYPE_NEOX;
    float freq_scale=1.f; size_t view_offset_bytes=0; bool factors=false, eligible=true, edge_positions=false;
};
static std::vector<case_spec> cases() {
    std::vector<case_spec> out;
    for(int h:{1,8}) for(int m:{11,264,268}) { case_spec s; s.heads=h;s.rows=m;
        s.name="rope_h"+std::to_string(h)+"_m"+std::to_string(m);out.push_back(s); }
    { case_spec s;s.name="rope_view_h8_m11";s.heads=8;s.view_offset_bytes=20;out.push_back(s); }
    { case_spec s;s.name="rope_position_edges_h1_m11";s.edge_positions=true;out.push_back(s); }
    { case_spec s;s.heads=8;s.name="rope_fallback_normal";s.rope_mode=0;s.eligible=false;out.push_back(s); }
    { case_spec s;s.heads=8;s.name="rope_fallback_head128";s.head_dim=128;s.eligible=false;out.push_back(s); }
    { case_spec s;s.heads=8;s.name="rope_fallback_rows2";s.rows=2;s.eligible=false;out.push_back(s); }
    { case_spec s;s.heads=8;s.name="rope_fallback_freqscale";s.freq_scale=.5f;s.eligible=false;out.push_back(s); }
    { case_spec s;s.heads=8;s.name="rope_fallback_factors";s.factors=true;s.eligible=false;out.push_back(s); }
    return out;
}
static void put_float(ggml_tensor & t,int index,float v) { std::memcpy(&t.op_params[index],&v,sizeof(v)); }
static void layout(ggml_tensor & t,ggml_type type,int64_t n0,int64_t n1,int64_t n2,int64_t n3,uintptr_t addr) {
    t={};t.type=type;t.ne[0]=n0;t.ne[1]=n1;t.ne[2]=n2;t.ne[3]=n3;
    size_t nb=(type==GGML_TYPE_F16?2:4);
    for(int i=0;i<4;++i) { t.nb[i]=nb;nb*=size_t(t.ne[i]); } t.data=reinterpret_cast<void *>(addr);
}
struct host_fixture {
    ggml_tensor src{},pos{},dst{},storage{},ff{};
    explicit host_fixture(const case_spec & s={}) {
        layout(src,GGML_TYPE_F32,s.head_dim,s.heads,s.rows,1,0x10000000u+s.view_offset_bytes);
        layout(pos,GGML_TYPE_I32,s.rows,1,1,1,0x20000000u);
        layout(dst,GGML_TYPE_F32,s.head_dim,s.heads,s.rows,1,0x30000000u);
        layout(ff,GGML_TYPE_F32,s.head_dim/2,1,1,1,0x40000000u);
        layout(storage,GGML_TYPE_F32,int64_t(size_t(s.head_dim)*s.heads*s.rows+s.view_offset_bytes/4+8),1,1,1,0x10000000u);
        if(s.view_offset_bytes) { src.view_src=&storage;src.view_offs=s.view_offset_bytes; }
        dst.op=GGML_OP_ROPE;dst.src[0]=&src;dst.src[1]=&pos;dst.src[2]=s.factors?&ff:nullptr;
        dst.op_params[1]=s.head_dim;dst.op_params[2]=s.rope_mode;dst.op_params[4]=2048;
        put_float(dst,5,10000.f);put_float(dst,6,s.freq_scale);put_float(dst,7,0.f);
        put_float(dst,8,1.f);put_float(dst,9,32.f);put_float(dst,10,1.f);
    }
    bool accepts() { ggml_flashrt_rope_plan plan{};return ggml_flashrt_plan_rope(&dst,plan); }
};
static int host_checks() {
    int n=0;auto check=[&](bool v,bool expected,const std::string & text){++n;need(v==expected,"host contract: "+text);};
    for(const auto & s:cases()) { host_fixture f(s);check(f.accepts(),s.eligible,s.name);
        if(s.eligible) { ggml_flashrt_rope_plan p{};const char *r="old";
            check(ggml_flashrt_plan_rope(&f.dst,p,&r)&&r==nullptr&&p.src==&f.src&&p.positions==&f.pos&&p.dst==&f.dst&&
                  p.heads==s.heads&&p.rows==s.rows&&p.tensor_bytes==size_t(s.head_dim)*s.heads*s.rows*4&&p.position_bytes==size_t(s.rows)*4,
                  true,"returned plan roles and extents"); }
    }
    {ggml_flashrt_rope_plan p{};const char *r=nullptr;check(ggml_flashrt_plan_rope(nullptr,p,&r),false,"null destination");check(r!=nullptr,true,"reason supplied");}
    {host_fixture f;f.dst.op=GGML_OP_ROPE_BACK;check(f.accepts(),false,"backward operator");}
    for(int which:{0,1}) {host_fixture f;f.dst.src[which]=nullptr;check(f.accepts(),false,"missing source");}
    for(int i=2;i<GGML_MAX_SRC;++i) {host_fixture f;f.dst.src[i]=&f.ff;check(f.accepts(),false,"extra source");}
    for(int which:{0,1,2}) {
        for(int d=0;d<4;++d) {
            {host_fixture f;auto*t=which==0?&f.src:which==1?&f.pos:&f.dst;t->nb[d]+=4;check(f.accepts(),false,"non-contiguous stride");}
            {host_fixture f;auto*t=which==0?&f.src:which==1?&f.pos:&f.dst;++t->ne[d];check(f.accepts(),false,"shape mismatch");}
        }
        {host_fixture f;auto*t=which==0?&f.src:which==1?&f.pos:&f.dst;t->data=nullptr;check(f.accepts(),false,"null data");}
        {host_fixture f;auto*t=which==0?&f.src:which==1?&f.pos:&f.dst;t->data=reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(t->data)+1);check(f.accepts(),false,"unaligned data");}
        {host_fixture f;auto*t=which==0?&f.src:which==1?&f.pos:&f.dst;t->data=reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()-3);check(f.accepts(),false,"address end overflow");}
        {host_fixture f;auto*t=which==0?&f.src:which==1?&f.pos:&f.dst;t->type=GGML_TYPE_F16;check(f.accepts(),false,"unsupported type");}
        {host_fixture f;auto*t=which==0?&f.src:which==1?&f.pos:&f.dst;t->ne[0]=std::numeric_limits<int64_t>::max();check(f.accepts(),false,"extent overflow");}
        {host_fixture f;auto*t=which==0?&f.src:which==1?&f.pos:&f.dst;t->view_offs=4;check(f.accepts(),false,"offset without view source");}
    }
    for(int which:{0,1,2}) {host_fixture f;
        if(which==0) f.dst.data=f.src.data;
        if(which==1) f.dst.data=f.pos.data;
        if(which==2) f.pos.data=f.src.data;
        check(f.accepts(),false,"pairwise alias");
    }
    {host_fixture f;f.dst.data=reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(f.src.data)+4);check(f.accepts(),false,"partial alias");}
    {host_fixture f;f.dst.data=reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(f.src.data)+256*11*4);check(f.accepts(),true,"adjacent ranges");}
    for(int index:{1,2,4}) {host_fixture f;f.dst.op_params[index]=-1;check(f.accepts(),false,"unsupported integer parameter");}
    for(int index=5;index<=10;++index) {
        {host_fixture f;put_float(f.dst,index,std::numeric_limits<float>::quiet_NaN());check(f.accepts(),false,"nonfinite float parameter");}
        {host_fixture f;float v=0;std::memcpy(&v,&f.dst.op_params[index],4);put_float(f.dst,index,v+1.f);check(f.accepts(),false,"unsupported float parameter");}
    }
    for(int index:{0,3}) {host_fixture f;f.dst.op_params[index]=12345;check(f.accepts(),true,"unused n_past or n_ctx ignored");}
    {case_spec s;s.view_offset_bytes=20;host_fixture f(s);f.src.data=reinterpret_cast<void*>(0x10000018u);check(f.accepts(),false,"resolved view address mismatch");}
    {case_spec s;s.view_offset_bytes=20;host_fixture f(s);f.storage.data=nullptr;check(f.accepts(),false,"view storage missing");}
    {case_spec s;s.view_offset_bytes=20;host_fixture f(s);f.src.view_offs=std::numeric_limits<size_t>::max()-3;check(f.accepts(),false,"view offset overflow");}
    std::cout<<n<<" RoPE metadata/address contract checks passed; aliases are host-only\n";return n;
}
#ifndef FLASHRT_ROPE_HOST_ONLY
// Direct double-to-binary16 round-to-nearest-even, independent of CUDA's cast.
static uint32_t round_even(double x) {
    double f=std::floor(x);uint32_t i=uint32_t(f);double rem=x-f;
    return i+uint32_t(rem>.5||(rem==.5&&(i&1)));
}
static uint16_t half_bits(double x) {
    const uint16_t sign=std::signbit(x)?0x8000u:0u;double a=std::fabs(x);
    if(std::isnan(a)) return uint16_t(sign|0x7e00u);
    if(std::isinf(a)||a>=65520.) return uint16_t(sign|0x7c00u);
    if(a<std::ldexp(1.,-14)) return uint16_t(sign|round_even(std::ldexp(a,24)));
    int e=0;std::frexp(a,&e);--e;uint32_t m=round_even(std::ldexp(a,10-e));
    if(m==2048) {++e;m=1024;}return uint16_t(sign|uint16_t((e+15)<<10)|uint16_t(m-1024));
}
static double from_half(uint16_t b) {
    const int e=(b>>10)&31;const int m=b&1023;
    double v=e==0?std::ldexp(double(m),-24):e==31?(m?std::numeric_limits<double>::quiet_NaN():std::numeric_limits<double>::infinity()):std::ldexp(double(1024+m),e-25);
    return b&0x8000u?-v:v;
}
static double half(double x) {return from_half(half_bits(x));}
static double half_step(double x) {
    const uint16_t b=uint16_t(half_bits(std::fabs(x))&0x7fffu);
    return from_half(uint16_t(b+1))-from_half(b);
}
static void reference_self_checks() {
    need(half_bits(1.)==0x3c00u && half_bits(-0.)==0x8000u, "reference one/signed zero");
    need(half_bits(1.00048828125)==0x3c00u && half_bits(1.00146484375)==0x3c02u, "reference normal ties-to-even");
    need(half_bits(std::ldexp(1.,-25))==0u && half_bits(3*std::ldexp(1.,-25))==2u, "reference subnormal ties-to-even");
    for (uint32_t bits=0;bits<65536;++bits) {
        if ((bits&0x7c00u)!=0x7c00u) {
            need(half_bits(from_half(uint16_t(bits)))==bits, "reference finite binary16 round trip");
        }
    }
}
static void env(const char*k,const char*v) {need((v?setenv(k,v,1):unsetenv(k))==0,k);}
static std::string quoted(const std::string & s) {
    std::ostringstream o;o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<char(c);else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<char(c);}o<<'"';return o.str();
}
static std::string fingerprint(const void *p,size_t bytes) {
    uint64_t h=14695981039346656037ull;const auto *b=static_cast<const unsigned char*>(p);
    for(size_t i=0;i<bytes;++i){h^=b[i];h*=1099511628211ull;}std::ostringstream o;o<<std::hex<<std::setw(16)<<std::setfill('0')<<h;return o.str();
}
static std::vector<int32_t> positions(const case_spec&s,bool changed) {
    std::vector<int32_t> p(size_t(s.rows));
    if(s.edge_positions) {
        const int32_t edge[]={-7,0,1,4094,4095,4096,8192,std::numeric_limits<int32_t>::min(),std::numeric_limits<int32_t>::max(),123,333};
        for(int i=0;i<s.rows;++i) p[size_t(i)]=edge[(i+(changed?3:0))%11];
    } else {for(int i=0;i<s.rows;++i) p[size_t(i)]=i+(changed?331:0);}
    return p;
}
static std::vector<float> input(const case_spec&s,bool changed) {
    std::vector<float> x(size_t(s.head_dim)*s.heads*s.rows);
    for(size_t i=0;i<x.size();++i){uint32_t h=uint32_t(i+7)*0x9e3779b1u^(changed?0xa47f9871u:0x329c16d7u);h^=h>>16;
        x[i]=float(int(h&0xffffffu)-8388608)/4194304.f;
        if(i%4093==0)x[i]=changed?-1.00048828125f:1.00146484375f;
    }return x;
}
static std::vector<float> factors(const case_spec&s) {
    std::vector<float> f(size_t(s.head_dim/2));for(size_t i=0;i<f.size();++i)f[i]=1.f+float(i%5)/16.f;return f;
}
struct gpu_fixture {
    ggml_context*ctx=nullptr;ggml_tensor *src=nullptr,*storage=nullptr,*pos=nullptr,*ff=nullptr,*dst=nullptr;ggml_cgraph*graph=nullptr;
    ggml_backend_buffer_t buffer=nullptr;case_spec spec;
    gpu_fixture(case_spec s,ggml_backend_t backend):spec(std::move(s)) {
        ctx=ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead_custom(16,false),nullptr,true});need(ctx,"ggml metadata allocation failed");
        if(spec.view_offset_bytes){storage=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,int64_t(size_t(spec.head_dim)*spec.heads*spec.rows+spec.view_offset_bytes/4+7));
            src=ggml_view_3d(ctx,storage,spec.head_dim,spec.heads,spec.rows,size_t(spec.head_dim)*4,size_t(spec.head_dim)*spec.heads*4,spec.view_offset_bytes);
        }else{src=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,spec.head_dim,spec.heads,spec.rows);storage=src;}
        pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,spec.rows);
        if(spec.factors)ff=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,spec.head_dim/2);
        dst=ggml_rope_ext(ctx,src,pos,ff,spec.head_dim,spec.rope_mode,2048,10000.f,spec.freq_scale,0.f,1.f,32.f,1.f);
        ggml_set_name(src,(spec.name+"_input").c_str());ggml_set_name(pos,(spec.name+"_positions").c_str());ggml_set_name(dst,spec.name.c_str());
        ggml_set_input(storage);ggml_set_input(pos);if(ff)ggml_set_input(ff);ggml_set_output(dst);
        graph=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(graph,dst);
        buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);need(buffer,"CUDA tensor allocation failed");ggml_backend_buffer_clear(buffer,0);
        ggml_flashrt_rope_plan p{};need(ggml_flashrt_plan_rope(dst,p)==spec.eligible,"allocated guard mismatch: "+spec.name);
        if(ff){auto f=factors(spec);ggml_backend_tensor_set(ff,f.data(),0,f.size()*4);}
    }
    ~gpu_fixture(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
};
struct reference_report {size_t elements=0,skipped=0;double max_abs=0,sum_sq=0,max_budget_ratio=0;};
static reference_report check_reference(const case_spec&s,const std::vector<float>&x,const std::vector<int32_t>&p,
                                         const std::vector<float>&actual,bool native,const std::string&label) {
    need(actual.size()==x.size(),label+": output extent mismatch");reference_report report;const auto f=factors(s);
    const float theta_scale=std::pow(10000.f,-2.f/float(s.head_dim));
    const double eps=std::numeric_limits<float>::epsilon();
    for(int row=0;row<s.rows;++row)for(int head=0;head<s.heads;++head)for(int pair=0;pair<s.head_dim/2;++pair){
        const size_t base=(size_t(row)*s.heads+head)*s.head_dim;
        const size_t a=base+size_t(s.rope_mode==GGML_ROPE_TYPE_NEOX?pair:2*pair);
        const size_t b=base+size_t(s.rope_mode==GGML_ROPE_TYPE_NEOX?pair+s.head_dim/2:2*pair+1);
        need(std::isfinite(actual[a])&&std::isfinite(actual[b]),label+": nonfinite output");
        // GGML's FP32 fast pow magnifies phase uncertainty at extreme I32
        // positions. The baseline edge rows retain finite/repeat/Graph checks,
        // but are explicitly excluded from a misleading host trig comparison.
        if(!native&&std::llabs(int64_t(p[size_t(row)]))>8192){report.skipped+=2;continue;}
        double x0=x[a],x1=x[b],c=0,si=0,theta=0;
        if(native){x0=half(x0);x1=half(x1);const double inv=1./std::pow(10000.,2.*pair/s.head_dim);
            theta=double(p[size_t(row)])*inv;c=half(std::cos(theta));si=half(std::sin(theta));
        }else{float th=float(p[size_t(row)])*std::pow(theta_scale,float(pair));
            if(s.factors) { th/=f[size_t(pair)]; } th*=s.freq_scale;theta=th;c=std::cos(th);si=std::sin(th);}
        const double exact[2]={x0*c-x1*si,x1*c+x0*si};const size_t indices[2]={a,b};
        for(int j=0;j<2;++j){const size_t i=indices[j];const double expected=native?half(exact[j]):exact[j];
            // Native: binary16 output rounding interval + bounded FP32 multiply/
            // FMA error. Cancellation uses product magnitudes, so a near-zero
            // result receives an absolute arithmetic allowance, not broad rtol.
            // Stored FP64 frequencies remove host/device pow discrepancies for
            // large positions; the 2e-11 term covers double sin/cos last bits.
            const double product_magnitude=std::fabs(x0*c)+std::fabs(x1*si)+std::fabs(x1*c)+std::fabs(x0*si);
            const double budget=native?(.5*half_step(expected)+3*eps*product_magnitude+2e-11*(std::fabs(x0)+std::fabs(x1))+1e-15):
                (8*eps*(1+std::fabs(theta))*(std::fabs(x0)+std::fabs(x1))+4*eps*product_magnitude+1e-12);
            const double distance=std::fabs(double(actual[i])-exact[j]);
            if(native){need(double(actual[i])==half(actual[i]),label+": candidate output is not representable as FP16");
                if(p[size_t(row)]==0)need(double(actual[i])==half(x[i]),label+": position zero must equal FP16-rounded input");}
            if(distance>budget){std::ostringstream err;err<<label<<": reference violation at "<<i<<" actual="<<std::setprecision(17)<<actual[i]<<" expected="<<expected<<" distance="<<distance<<" budget="<<budget<<" position="<<p[size_t(row)];throw std::runtime_error(err.str());}
            const double error=double(actual[i])-expected;++report.elements;report.max_abs=std::max(report.max_abs,std::fabs(error));report.sum_sq+=error*error;report.max_budget_ratio=std::max(report.max_budget_ratio,distance/budget);
        }
    }return report;
}
struct device_owner {ggml_backend_t b=nullptr;~device_owner(){if(b){ggml_backend_synchronize(b);ggml_backend_free(b);}}};
static void gpu_tests(const options&o,int host_count){
    reference_self_checks();
    env("GGML_CUDA_FLASHRT","1");env("GGML_CUDA_FLASHRT_ROPE",o.mode?"1":"0");env("GGML_CUDA_FLASHRT_TRACE","1");env("GGML_CUDA_DISABLE_GRAPHS",o.graphs?nullptr:"1");
    device_owner device;device.b=ggml_backend_cuda_init(0);need(device.b,"CUDA0 initialization failed");
    const uint16_t endian=1;need(*reinterpret_cast<const unsigned char*>(&endian)==1,"raw outputs require little-endian host");
    std::ofstream raw(o.prefix+".bin",std::ios::binary|std::ios::trunc);need(bool(raw),"cannot open raw output");
    std::ostringstream records;records<<std::setprecision(17);size_t offset=0;bool first=true;int record_count=0;
    for(const auto&s:cases()){
        gpu_fixture f(s,device.b);std::vector<float> previous,storage_after;
        for(int phase_index=0;phase_index<3;++phase_index){
            const char*phase=phase_index==0?"initial":phase_index==1?"positions_changed":"input_changed";
            const auto x=input(s,phase_index==2);const auto p=positions(s,phase_index!=0);const int calls=phase_index==0?8:3;
            std::vector<float> storage(size_t(ggml_nelements(f.storage)),-777.25f);
            std::copy(x.begin(),x.end(),storage.begin()+s.view_offset_bytes/4);
            // The position phase writes exactly the original input again. The
            // input phase writes exactly the latest positions again. Both use
            // the very same graph and allocations across all three phases.
            ggml_backend_tensor_set(f.storage,storage.data(),0,storage.size()*4);ggml_backend_tensor_set(f.pos,p.data(),0,p.size()*4);
            std::vector<float> actual(x.size()),initial;reference_report ref;
            for(int call=0;call<calls;++call){need(ggml_backend_graph_compute(device.b,f.graph)==GGML_STATUS_SUCCESS,s.name+": graph compute failed");ggml_backend_synchronize(device.b);
                ggml_backend_tensor_get(f.dst,actual.data(),0,actual.size()*4);ref=check_reference(s,x,p,actual,o.mode&&s.eligible,s.name+"/"+phase);
                if(call==0)initial=actual;else need(std::memcmp(initial.data(),actual.data(),actual.size()*4)==0,s.name+": repeated output not byte exact");
            }
            if(phase_index) { need(std::memcmp(previous.data(),actual.data(),actual.size()*4)!=0,s.name+": mutated input or positions did not affect output"); } previous=actual;
            storage_after.resize(storage.size());ggml_backend_tensor_get(f.storage,storage_after.data(),0,storage_after.size()*4);
            need(std::memcmp(storage_after.data(),storage.data(),storage.size()*4)==0,s.name+": source storage/guards modified");
            std::vector<int32_t>pos_after(p.size());ggml_backend_tensor_get(f.pos,pos_after.data(),0,pos_after.size()*4);need(pos_after==p,s.name+": positions overwritten");
            raw.write(reinterpret_cast<const char*>(actual.data()),std::streamsize(actual.size()*4));need(bool(raw),"raw output write failed");
            if(!first) { records<<",\n"; } first=false;++record_count;
            records<<"{\"name\":"<<quoted(s.name)<<",\"node\":"<<quoted(s.name)<<",\"mode\":"<<o.mode<<",\"head_dim\":"<<s.head_dim<<",\"heads\":"<<s.heads<<",\"rows\":"<<s.rows
                   <<",\"rope_mode\":"<<s.rope_mode<<",\"freq_scale\":"<<s.freq_scale<<",\"has_freq_factors\":"<<(s.factors?"true":"false")<<",\"view_offset_bytes\":"<<s.view_offset_bytes
                   <<",\"eligible\":"<<(s.eligible?"true":"false")<<",\"expected_candidate\":"<<((s.eligible&&o.mode)?"true":"false")<<",\"phase\":"<<quoted(phase)
                   <<",\"raw_dtype\":\"little-endian-f32\",\"raw_offset_bytes\":"<<offset<<",\"raw_bytes\":"<<actual.size()*4<<",\"elements\":"<<actual.size()<<",\"compute_calls\":"<<calls
                   <<",\"finite\":true,\"reference_kind\":"<<quoted((s.eligible&&o.mode)?"flashrt_fp16":"ggml_fp32_fastmath")<<",\"reference_budget_version\":\"rope-fp16-rounding-v1\""
                   <<",\"reference_elements\":"<<ref.elements<<",\"reference_skipped_elements\":"<<ref.skipped<<",\"reference_max_abs\":"<<ref.max_abs<<",\"reference_rmse\":"<<std::sqrt(ref.sum_sq/std::max<size_t>(1,ref.elements))
                   <<",\"reference_max_budget_ratio\":"<<ref.max_budget_ratio<<",\"reference_violations\":0,\"repeat_bit_mismatches\":0,\"input_fingerprint\":"<<quoted(fingerprint(x.data(),x.size()*4))
                   <<",\"position_fingerprint\":"<<quoted(fingerprint(p.data(),p.size()*4))<<",\"input_changed\":"<<(phase_index==2?"true":"false")<<",\"positions_changed\":"<<(phase_index!=0?"true":"false")
                   <<",\"position_values\":[";
            for(size_t i=0;i<p.size();++i){if(i)records<<',';records<<p[i];}records<<"]}";offset+=actual.size()*4;
            std::cout<<"ROPE_CASE name="<<s.name<<" phase="<<phase<<" expected_candidate="<<int(s.eligible&&o.mode)<<" reference_elements="<<ref.elements<<" skipped="<<ref.skipped<<" max_abs="<<ref.max_abs<<" repeat_exact=1\n";
        }
    }
    raw.close();need(bool(raw),"raw close failed");std::ofstream json(o.prefix+".json",std::ios::trunc);need(bool(json),"cannot open manifest");
    json<<"{\"schema_version\":1,\"status\":\"tests_complete\",\"backend\":\"CUDA0\",\"mode\":"<<o.mode<<",\"graphs_requested\":"<<(o.graphs?"true":"false")<<",\"host_checks\":"<<host_count
        <<",\"case_count\":"<<cases().size()<<",\"record_count\":"<<record_count<<",\"raw_file\":"<<quoted(o.prefix+".bin")<<",\"raw_bytes\":"<<offset
        <<",\"input_update_checked\":true,\"position_update_checked\":true,\"gpu_alias_cases\":0,\"reference_budget_version\":\"rope-fp16-rounding-v1\""
        <<",\"reference\":\"Independent split-half/adjacent indexing, direct double-to-half RNE, FP16 input/coefficient/output rounding; GGML baseline checked separately with FP32 phase error budget\""
        <<",\"reference_scope\":\"All candidate elements every compute; baseline only extreme I32 rows beyond abs(position)=8192 skip host trig, with finite/repeat/Graph checks retained\""
        <<",\"comparison_policy\":\"Same-mode Graph settings and all fallback outputs must match raw bytes; candidate/native precision is compared with a quantization-aware per-element bound\""
        <<",\"records\":[\n"<<records.str()<<"\n]}\n";json.close();need(bool(json),"manifest close failed");std::cout<<"CUDA RoPE tests passed: "<<o.prefix<<".json\n";
}
#endif
} // namespace
int main(int argc,char**argv){try{const auto o=parse(argc,argv);const int n=host_checks();
#ifdef FLASHRT_ROPE_HOST_ONLY
(void)n;need(o.host_only,"this binary was compiled host-only");
#else
if(!o.host_only)gpu_tests(o,n);
#endif
return 0;}catch(const std::exception&e){std::cerr<<"RoPE test failed: "<<e.what()<<'\n';return 1;}}
