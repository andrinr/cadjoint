import{S as W,F as ne,E as T,r as F,t as ue,f as I,s as b,A as q,U as le,p as z,m as ce,a as D,i as R,b as Y,V as N,c as de,e as pe}from"./jaxjs-Bp-xEarF.js";import"./index-BI5YHb4Z.js";const fe=`
fn threefry2x32(key: vec2<u32>, ctr: vec2<u32>) -> vec2<u32> {
  let ks0: u32 = key.x;
  let ks1: u32 = key.y;
  let ks2: u32 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  var x0: u32 = ctr.x + ks0;
  var x1: u32 = ctr.y + ks1;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 1u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 2u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks0;
  x1 += ks1 + 3u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 4u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 5u;

  return vec2<u32>(x0, x1);
}`,me=`
const _erf_p: f32 = 0.3275911;
const _erf_a1: f32 = 0.254829592;
const _erf_a2: f32 = -0.284496736;
const _erf_a3: f32 = 1.421413741;
const _erf_a4: f32 = -1.453152027;
const _erf_a5: f32 = 1.061405429;
fn erf(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  return sign(x) * (1.0 - P_t * exp(-x * x));
}
fn erfc(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  let E = P_t * exp(-x * x);
  return select(2.0 - E, E, x >= 0.0);
}`,O=String.raw`
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn inf() -> f32 { let bits = 0x7f800000u; return bitcast<f32>(bits); }
`.trim();var te=class{pushIndent=Symbol("pushIndent");popIndent=Symbol("popIndent");lines=[];#e="";emit(...t){for(const e of t)e===this.pushIndent?this.#e+="  ":e===this.popIndent?this.#e=this.#e.slice(0,-2):this.lines.push(e?this.#e+e:"")}emitPreamble(t,e){let i=!1,a=new Map;for(const o of e)o!=null&&(i||=o.some(r=>r.dtype==="float16"),a=ce(a,o.distinctOps()));if(i){if(!t.features.has("shader-f16"))throw new Error("WebGPU device does not support shader-f16 feature");this.emit("enable f16;")}this.emit(O),a.has("Threefry2x32")&&this.emit(fe),(a.has("Erf")||a.has("Erfc"))&&this.emit(me),this.emit("")}emitPhonyAssignments(t){t.length>0&&this.emit(t.map(e=>`_ = &${e};`).join(" "))}toString(){return this.lines.join(`
`)}};function y(t,e=!1){switch(t){case"bool":return e?"i32":"bool";case"int32":return"i32";case"uint32":return"u32";case"float32":return"f32";case"float16":return"f16";default:throw new Error(`Unsupported dtype for WebGPU: ${t}`)}}function re(t){switch(t){case"bool":return"1";case"int32":return"2147483647";case"uint32":return"4294967295u";case"float32":return"inf()";case"float16":return"f16(inf())";default:throw new Error(`Unsupported dtype for WebGPU: ${t}`)}}function ie(t,e){if(t==="bool")return e?"true":"false";if(t==="int32")return e.toString();if(t==="uint32")return e.toString()+"u";if(t==="float32")return Number.isNaN(e)?"nan()":Number.isFinite(e)?"f32("+e.toString()+")":e>0?"inf()":"-inf()";if(t==="float16")return Number.isNaN(e)?"f16(nan())":Number.isFinite(e)?"f16("+e.toString()+")":e>0?"f16(inf())":"f16(-inf())";throw new Error(`Unsupported const dtype: ${t}`)}function _e(t,e,i,a){if(t==="Add")return`(${i} + ${a})`;if(t==="Mul")return`(${i} * ${a})`;if(t==="Min")return e==="bool"?`(${i} && ${a})`:`min(${i}, ${a})`;if(t==="Max")return e==="bool"?`(${i} || ${a})`:`max(${i}, ${a})`;throw new Error(`Unsupported reduction op: ${t}`)}var ae=class{wb;args;#e=0;#t=new Map;#i=new Set;#a=new Map;constructor(t,e){this.wb=t,this.args=e}#r(){return`alu${this.#e++}`}#s(t){return t.match(/^alu[0-9]+$/)}countReferences(t){if(this.#t.set(t,(this.#t.get(t)??0)+1),!this.#i.has(t)){this.#i.add(t);for(const e of t.src)this.countReferences(e)}}reset(){this.#t.clear(),this.#i.clear(),this.#a.clear()}run(t){if(this.#a.has(t))return this.#a.get(t);const{op:e,src:i,dtype:a,arg:o}=t;let r="";if(D.Binary.has(e)||D.Compare.has(e)){const s=this.run(i[0]),n=this.run(i[1]);if(e==="Add")a==="bool"?r=`(${s} || ${n})`:r=`(${s} + ${n})`;else if(e==="Sub")r=`(${s} - ${n})`;else if(e==="Mul")a==="bool"?r=`(${s} && ${n})`:r=`(${s} * ${n})`;else if(e==="Idiv")r=R(a)?`trunc(${s} / ${n})`:`(${s} / ${n})`;else if(e==="Mod")r=`(${s} % ${n})`;else if(e==="Min")a==="bool"?r=`(${s} && ${n})`:r=`min(${b(s)}, ${b(n)})`;else if(e==="Max")a==="bool"?r=`(${s} || ${n})`:r=`max(${b(s)}, ${b(n)})`;else if(e==="BitCombine")o==="and"?r=`(${s} & ${n})`:o==="or"?r=`(${s} | ${n})`:r=a==="bool"?`(${s} != ${n})`:`(${s} ^ ${n})`;else if(e==="BitShift")o==="shl"?r=`(${s} << ${n})`:r=`(${s} >> ${n})`;else if(e==="Cmplt")r=`(${s} < ${n})`;else if(e==="Cmpne")if(R(i[0].dtype)){const p=this.#s(s)?s:this.#r();p!==s&&this.wb.emit(`let ${p} = ${s};`),r=`(${p} != ${n} || min(${p}, ${y(i[0].dtype)}(inf())) != ${p})`}else r=`(${s} != ${n})`}else if(D.Unary.has(e))if(e==="Reciprocal"&&i[0].op==="Sqrt")r=`inverseSqrt(${this.run(i[0].src[0])})`;else{const s=this.run(i[0]);if(e==="Sin")r=`sin(${b(s)})`;else if(e==="Cos")r=`cos(${b(s)})`;else if(e==="Asin")r=`asin(${b(s)})`;else if(e==="Atan")r=`atan(${b(s)})`;else if(e==="Exp")r=`exp(${b(s)})`;else if(e==="Log")r=`log(${b(s)})`;else if(e==="Erf"||e==="Erfc"){const n=e==="Erf"?"erf":"erfc";a!=="float32"?r=`${y(a)}(${n}(f32(${b(s)})))`:r=`${n}(${b(s)})`}else if(e==="Sqrt")r=`sqrt(${b(s)})`;else if(e==="Reciprocal")r=`(1.0 / ${s})`;else if(e==="Floor")r=`floor(${b(s)})`;else if(e==="Ceil")r=`ceil(${b(s)})`;else if(e==="Cast"){const n=y(i[0].dtype),p=y(a);if(R(i[0].dtype)&&!(R(a)||a==="bool")){const d=re(a),m=this.#s(s)?s:this.#r();m!==s&&this.wb.emit(`let ${m}: ${n} = ${b(s)};`),r=`select(${p}(${m}), ${d}, ${m} >= ${n}(${d}))`}else r=`${p}(${b(s)})`}else e==="Bitcast"?r=`bitcast<${y(a)}>(${b(s)})`:e==="BitCount"&&(r=`i32(countOneBits(${b(s)}))`)}else if(e==="Where")r=`select(${b(this.run(i[2]))}, ${b(this.run(i[1]))}, ${b(this.run(i[0]))})`;else if(e==="Threefry2x32"){const s=this.#r(),[n,p,d,m]=i.map(w=>b(this.run(w)));if(this.wb.emit(`let ${s} = threefry2x32(vec2(${n}, ${p}), vec2(${d}, ${m}));`),o==="xor")r=`(${s}.x ^ ${s}.y)`;else if(o===0)r=`${s}.x`;else if(o===1)r=`${s}.y`;else throw new Y(e,a,"webgpu",o)}else{if(e==="Const")return ie(a,o);if(e==="Special")return o[0];if(e==="Variable")return o;e==="GlobalIndex"&&(r=`${this.args[o[0]]}[${b(this.run(i[0]))}]`,a==="bool"&&(r=`(${r} != 0)`))}if(!r)throw new Y(e,a,"webgpu",o);const l=y(a);if((this.#t.get(t)??0)>1){const s=this.#r();return this.#a.set(t,s),this.wb.emit(`let ${s}: ${l} = ${b(r)};`),s}else return this.#a.set(t,r),r}};const B=16384;function U(t){let e=t,i=1;return t>65535&&(e=B,i=Math.ceil(t/B)),[e,i]}const ge=1,xe=20,he=5,be=1e3,$e=typeof navigator<"u"&&navigator.userAgent.includes("Firefox")&&!navigator.userAgent.includes("Seamonkey");let we;const K=[];function V(){const t=(we??=(()=>{const e=new MessageChannel;return e.port1.onmessage=()=>K.shift()?.(),e.port1.start(),e})()).port2;return new Promise(e=>{K.push(e),t.postMessage(0)})}async function ve(t,e){const i=e.mapAsync(GPUMapMode.READ);if(!$e)return i;let a=!1;const o=i.finally(()=>{a=!0});await V();const r=performance.now();let l=-1/0;for(;!a;){const s=performance.now()-r;if(s>=be)break;performance.now()-l>=ge&&(l=performance.now(),t.queue.submit([t.createCommandEncoder().finish()])),s<xe?await V():await new Promise(n=>setTimeout(n,he))}return o}function ye(t){return t==="float16"?"float32":t==="bool"?"int32":t}function ke({name:t,dtype:e,uniformDtype:i}){const a=q.variable(i,`uniforms.${t}`);return e==="float16"?q.cast("float16",a):e==="bool"?q.cmpne(a,q.i32(0)):a}function Se(t,e,i,a){switch(i){case"float32":t.setFloat32(e,a,!0);break;case"int32":t.setInt32(e,a,!0);break;case"uint32":t.setUint32(e,a,!0);break;default:throw new Error(`Unsupported dtype for constant uniform: ${i}`)}}function je(t){const e=[];return[t.rewrite(i=>{if(i.op!=="Const"||i.arg===0)return;const a={name:`c${e.length}`,dtype:i.dtype,uniformDtype:ye(i.dtype),value:i.arg};return e.push(a),ke(a)}),e]}function Ue(t){const e=new Uint8Array(t.length*4),i=new DataView(e.buffer);return t.forEach((a,o)=>Se(i,o*4,a.uniformDtype,a.value)),e}function Be(t,e){if(e.nargs!==0||e.reduction)return null;let i=e.exp.substitute({gidx:q.special("int32","gidx",e.size)}).simplify(),a=[];[i,a]=je(i);const o=new te;o.emitPreamble(t,[i]),a.length>0&&o.emit("struct Uniforms {",o.pushIndent,...a.map(m=>`${m.name}: ${y(m.uniformDtype)},`),o.popIndent,`}
`);const r=y(e.dtype,!0);o.emit(`@group(0) @binding(0) var<storage, read_write> result : array<${r}>;`),a.length>0&&o.emit("@group(1) @binding(0) var<uniform> uniforms: Uniforms;");const l=I(e.size,256),[s,n]=U(Math.ceil(e.size/l));if(o.emit("",`@compute @workgroup_size(${l})`,"fn main(@builtin(global_invocation_id) id : vec3<u32>) {",o.pushIndent),n===1)o.emit(`if (id.x >= ${e.size}) { return; }`,"let gidx: i32 = i32(id.x);");else{const m=s*l;o.emit(`if (${m} * id.y + id.x >= ${e.size}) { return; }`,`let gidx: i32 = i32(${m} * id.y + id.x);`)}const p=new ae(o,[]);p.countReferences(i);let d=b(p.run(i));return r!==y(i.dtype)&&(d=`${r}(${d})`),o.emit(`result[gidx] = ${d};`,o.popIndent,"}"),{code:o.toString(),numInputs:0,numOutputs:1,hasUniform:a.length>0,passes:[{grid:[s,n],uniform:a.length>0?Ue(a):void 0}]}}var Ee=class S{device;static alphaModes=["opaque","premultiplied"];static width=256;static height=256;initialized=!1;deviceStorage;deviceContexts;hostStorage;hostContext;constructor(e){this.device=e}#e(){if(typeof OffscreenCanvas>"u")throw new Error("OffscreenCanvas is not available in this environment, so you cannot read data from WebGPU synchronously. Consider using the async API.");const e=()=>new OffscreenCanvas(S.width,S.height);this.deviceStorage=S.alphaModes.map(e),this.deviceContexts=this.deviceStorage.map((i,a)=>{const o=i.getContext("webgpu");return o.configure({device:this.device,format:"bgra8unorm",usage:GPUTextureUsage.COPY_DST,alphaMode:S.alphaModes[a]}),o}),this.hostStorage=e(),this.hostContext=this.hostStorage.getContext("2d",{willReadFrequently:!0}),this.initialized=!0}read(e,i,a){this.initialized||this.#e();const o=this.deviceStorage,r=this.deviceContexts,l=this.hostContext,s=Math.ceil(a/4),n=S.width*4,p=new ArrayBuffer(s*4);for(let d=0;d<r.length;d++){const m=r[d].getCurrentTexture(),w=(x,v,M)=>{const A=this.device.createCommandEncoder();A.copyBufferToTexture({buffer:e,bytesPerRow:n,offset:M+i},{texture:m},{width:x,height:v,depthOrArrayLayers:1});const E=A.finish();this.device.queue.submit([E]),l.clearRect(0,0,x,v),l.drawImage(o[d],0,0);const C=l.getImageData(0,0,x,v).data,j=new Uint8ClampedArray(p,M,4*x*v),u=S.alphaModes[d];for(let c=0;c<j.length;c+=4)u==="premultiplied"?j[c+3]=C[c+3]:(j[c]=C[c+2],j[c+1]=C[c+1],j[c+2]=C[c])},$=S.width*S.height,_=Math.floor(s/$);let g=s%$;const h=Math.floor(g/S.width);g=g%S.width;let f=0;for(let x=0;x<_;x++)w(S.width,S.height,f),f+=$*4;h>0&&(w(S.width,h,f),f+=h*S.width*4),g>0&&w(g,1,f)}return new Uint8Array(p,0,a)}};function Ce(t){const e=new Uint32Array(3);return e[0]=t.kind==="sort"?0:1,e[1]=t.mergeStep??0,e[2]=t.mergeStage??0,new Uint8Array(e.buffer)}function se(t,e,i,a,o){const r=y(e,!0),l=1<<Math.ceil(Math.log2(i||1)),s=Math.ceil(l/2),n=I(s,t.limits.maxComputeWorkgroupSizeX),p=s/n,d=Math.log2(l),m=Math.min(d,Math.log2(n*2)),w=e==="float16",$=R(e)?`${r}(nan())`:re(e),_=`
${w?"enable f16;":""}
${O}

struct Uniforms {
  kind: u32, // 0 = sort, 1 = merge
  merge_step: u32, // half_block = 2^step
  merge_stage: u32, // only used for merge
}

@group(0) @binding(0) var<storage, read> input: array<${r}>;
@group(0) @binding(1) var<storage, read_write> output: array<${r}>;
${o?"@group(0) @binding(2) var<storage, read_write> output_idx: array<i32>;":""}

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

var<workgroup> shared_vals: array<${r}, ${n*2}>;
${o?`var<workgroup> shared_idx: array<i32, ${n*2}>;`:""}

fn compare(a: ${r}, b: ${r}) -> bool {
${R(e)?`
  let min_value = min(a, b);
  return a == min_value && b != min_value;`:"  return a < b;"}
}

fn compare_and_swap(i: u32, j: u32) {
  let val_i = shared_vals[i];
  let val_j = shared_vals[j];
${o?`
  if (
    compare(val_j, val_i) ||
    (!compare(val_i, val_j) && shared_idx[j] < shared_idx[i])
  ) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
    let tmp_idx = shared_idx[i];
    shared_idx[i] = shared_idx[j];
    shared_idx[j] = tmp_idx;
  }`:`
  if (compare(val_j, val_i)) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
  }`}
}

@compute @workgroup_size(${n})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let blockid = wg_id.x + wg_id.y * ${B}u;
  let batch = blockid / ${p}u;
  let wg_in_batch = blockid % ${p}u;

  let tid = local_id.x;
  let base = batch * ${i}u;

  if (uniforms.kind == 0u || (uniforms.kind == 1u && uniforms.merge_step == ${m-1}u)) {
    let wg_base = wg_in_batch * ${n*2}u;

    // Load data into shared memory (2 elements per thread)
    let idx0 = tid * 2u;
    let idx1 = tid * 2u + 1u;
    // Load from input for initial 'sort' pass, then from output (read-write) for 'merge' passes.
    if (uniforms.kind == 0u) {
      shared_vals[idx0] = select(${$}, input[base + wg_base + idx0], wg_base + idx0 < ${i}u);
      shared_vals[idx1] = select(${$}, input[base + wg_base + idx1], wg_base + idx1 < ${i}u);
${o?`
      shared_idx[idx0] = i32(wg_base + idx0);
      shared_idx[idx1] = i32(wg_base + idx1);`:""}
    } else {
      shared_vals[idx0] = select(${$}, output[base + wg_base + idx0], wg_base + idx0 < ${i}u);
      shared_vals[idx1] = select(${$}, output[base + wg_base + idx1], wg_base + idx1 < ${i}u);
${o?`
      shared_idx[idx0] = select(${i}, output_idx[base + wg_base + idx0], wg_base + idx0 < ${i}u);
      shared_idx[idx1] = select(${i}, output_idx[base + wg_base + idx1], wg_base + idx1 < ${i}u);`:""}
    }
    workgroupBarrier();

    let initial_stage = select(0u, ${m-1}u, uniforms.kind != 0u);
    for (var stage = initial_stage; stage < ${m}u; stage++) {
      for (var step1 = stage + 1u; step1 > 0u; step1--) {
        let step = step1 - 1u;
        let half_block = 1u << step;
        let is_first_step = uniforms.kind == 0u && step == stage;

        let block_offset = (tid / half_block) * half_block;
        let local_offset = tid % half_block;
        let i = block_offset * 2u + local_offset;
        let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);
        compare_and_swap(i, j);

        workgroupBarrier();
      }
    }

    if (wg_base + idx0 < ${i}u) {
      output[base + wg_base + idx0] = shared_vals[idx0];
      ${o?"output_idx[base + wg_base + idx0] = shared_idx[idx0];":""}
    }
    if (wg_base + idx1 < ${i}u) {
      output[base + wg_base + idx1] = shared_vals[idx1];
      ${o?"output_idx[base + wg_base + idx1] = shared_idx[idx1];":""}
    }
  } else {
    // Execute single merge pass for a step >= numLocalStages.
    let half_block = 1u << uniforms.merge_step;  // half_block >= workgroupSize * 2
    let thread_in_batch = wg_in_batch * ${n} + tid;
    let is_first_step = uniforms.merge_step == uniforms.merge_stage;

    let block_offset = (thread_in_batch / half_block) * half_block;
    let local_offset = thread_in_batch % half_block;
    let i = block_offset * 2u + local_offset;
    let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);

    // Global version of compare_and_swap()
    if (j < ${i}u) {
      let val_i = output[base + i];
      let val_j = output[base + j];
${o?`
      let idx_i = output_idx[base + i];
      let idx_j = output_idx[base + j];
      if (compare(val_j, val_i) || (!compare(val_i, val_j) && idx_j < idx_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;
        output_idx[base + i] = idx_j;
        output_idx[base + j] = idx_i;`:`
      if (compare(val_j, val_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;`}
      }
    }
  }
}
`.trim(),g=U(a*p),h=[{kind:"sort"}];for(let f=m;f<d;f++)for(let x=f;x>=m-1;x--)h.push({kind:"merge",mergeStep:x,mergeStage:f});return[{code:_,numInputs:1,numOutputs:o?2:1,hasUniform:!0,passes:h.map(f=>({grid:g,uniform:Ce(f)}))}]}function Me(t,e){const i=e.inputDtypes[0],a=e.inputShapes[0],o=a[a.length-1];return se(t,i,o,z(a.slice(0,-1)),!1)}function Pe(t,e){const i=e.inputDtypes[0],a=e.inputShapes[0],o=a[a.length-1];return se(t,i,o,z(a.slice(0,-1)),!0)}function oe(t,e,i){if(e.shape[i]<=1)return"0u";const a=e.strides[i];return`(${a===1?t:`(${t} / ${a}u)`} % ${e.shape[i]}u)`}function qe(t,e,i,a,o){const r=a+o-i.ndim,l=[];for(let s=0;s<i.ndim;s++){const n=i.strides[s];if(n===0)continue;const p=oe(t,e,r+s);l.push(n===1?p:`(${p} * ${n}u)`)}return l.length===0?"0u":l.join(" + ")}function ze(t){switch(t){case"float32":return{statement:"atomic_add_f32(&output[output_index], updates[global]);",helpers:`
fn atomic_add_f32(address: ptr<storage, atomic<u32>, read_write>, value: f32) {
  var old_bits = atomicLoad(address);
  loop {
    let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
    let result = atomicCompareExchangeWeak(address, old_bits, new_bits);
    if (result.exchanged) {
      break;
    }
    old_bits = result.old_value;
  }
}`};case"float16":return{statement:`
  atomic_add_f16(
    &output[output_index / 2u],
    output_index % 2u != 0u,
    updates[global],
  );`.trim(),helpers:`
fn atomic_add_f16(
  address: ptr<storage, atomic<u32>, read_write>,
  high: bool,
  value: f16,
) {
  var old_bits = atomicLoad(address);
  loop {
    var pair = unpack2x16float(old_bits);
    if (high) {
      pair.y += f32(value);
    } else {
      pair.x += f32(value);
    }
    let new_bits = pack2x16float(pair);
    let result = atomicCompareExchangeWeak(address, old_bits, new_bits);
    if (result.exchanged) {
      break;
    }
    old_bits = result.old_value;
  }
}`};case"uint32":case"int32":return{statement:"atomicAdd(&output[output_index], updates[global]);",helpers:""};case"bool":return{statement:"atomicOr(&output[output_index], updates[global]);",helpers:""};default:throw new Error(`Unsupported atomic Scatter dtype for WebGPU: ${t}`)}}function Ie(t,e,{op:i,shape:a,axis:o,outDim:r,uniqueIndices:l}){const s=e.inputDtypes[0],n=N.create(e.inputShapes[0]),p=e.inputShapes.slice(1).map(u=>N.create(u)),d=Math.max(...p.map(u=>u.ndim)),m=n.size,w=y(s,!0),$=s==="float16";if($&&!t.features.has("shader-f16"))throw new Error("WebGPU device does not support shader-f16 feature");const _=[...F(r),...F(r+d,n.ndim)],g=p.map((u,c)=>`  let scatter_index_${c} = i32(index_${c}[${qe("global",n,u,r,d)}]);`),h=o.map((u,c)=>`scatter_index_${c} < 0 || scatter_index_${c} >= ${a[u]}`),f=N.create(a);let x=0;const v=F(a.length).map(u=>{const c=o.indexOf(u);let k;return c===-1?k=oe("global",n,_[x++]):k=`u32(scatter_index_${c})`,f.strides[u]===1?k:`(${k} * ${f.strides[u]}u)`}).join(" + "),M=i==="add"&&!l,A=M?`atomic<${s==="int32"||s==="bool"?"i32":"u32"}>`:w,E=M?ze(s):{statement:"output[output_index] = updates[global];",helpers:""},C=t.limits.maxComputeWorkgroupSizeX,j=I(Math.min(Math.max(m,1),C),C);return[{code:`
${$?"enable f16;":""}
${O}
${E.helpers}

@group(0) @binding(0) var<storage, read> updates: array<${w}>;
${p.map((u,c)=>`@group(0) @binding(${c+1}) var<storage, read> index_${c}: array<${y(e.inputDtypes[c+1],!0)}>;`).join(`
`)}
@group(0) @binding(${e.inputShapes.length}) var<storage, read_write> output: array<${A}>;

@compute @workgroup_size(${j})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global = global_id.x + global_id.y * ${B*j}u;
  if (global >= ${m}u) {
    return;
  }
${g.join(`
`)}
  if (${h.join(" || ")}) {
    return;
  }
  let output_index = ${v};
  ${E.statement}
}
`.trim(),numInputs:e.inputShapes.length,numOutputs:1,hasUniform:!1,clearOutputs:!0,passes:[{grid:U(Math.ceil(m/j))}]}]}function Ae(t,e,i){const a=e.inputDtypes[0],o=e.inputShapes[0],r=e.inputShapes[1],l=o[o.length-1],s=r[r.length-2],n=z(o.slice(0,-2)),p=a==="float16",d=y(a,!0),m=I(l,t.limits.maxComputeWorkgroupSizeX);return[{code:`
${p?"enable f16;":""}
${O}

@group(0) @binding(0) var<storage, read> a: array<${d}>;
@group(0) @binding(1) var<storage, read> b: array<${d}>;
@group(0) @binding(2) var<storage, read_write> x: array<${d}>;

// Shared memory for the current pivot value x[j]
var<workgroup> x_j: ${d};

@compute @workgroup_size(${m})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let wg_idx = wg_id.x + wg_id.y * ${B}u;
  let mat_idx = wg_idx / ${s}u;
  let rhs_idx = wg_idx % ${s}u;

  if (mat_idx >= ${n}u) {
    return;
  }

  let a_base = mat_idx * ${l*l}u;
  let bx_base = (mat_idx * ${s}u + rhs_idx) * ${l}u;
  let tid = local_id.x;

  // Step 1: Copy b to x (threads collaborate)
  for (var idx = tid; idx < ${l}u; idx += ${m}u) {
    x[bx_base + idx] = b[bx_base + idx];
  }
  storageBarrier();

  // Step 2: Back-substitution from j = n-1 down to 0
  for (var jj = 0u; jj < ${l}u; jj++) {
    let j = ${l-1}u - jj;

    // Thread 0 computes x[j] = x[j] / a[j,j]
    if (tid == 0u) {
      ${i.unitDiagonal?"x_j = x[bx_base + j];":`x_j = x[bx_base + j] / a[a_base + j * ${l}u + j];`}
      x[bx_base + j] = x_j;
    }
    workgroupBarrier();  // Sync shared memory x_j

    // All threads subtract x[j] * a[i,j] from x[i] for i < j
    for (var i = tid; i < j; i += ${m}u) {
      x[bx_base + i] -= x_j * a[a_base + i * ${l}u + j];
    }
    workgroupBarrier();
    storageBarrier();
  }
}
`.trim(),numInputs:2,numOutputs:1,hasUniform:!1,passes:[{grid:U(n*s)}]}]}const P={Unblocked:0,InitBlocked:1,FactorBlock:2,SolvePanel:3,UpdateTrailing:4},Le=16,Oe=256,L=16;function G(t,e,i,a){const o=new Uint32Array(4);return o[0]=t,o[1]=e,o[2]=i,o[3]=a,new Uint8Array(o.buffer)}function Re(t,e){const i=e.inputDtypes[0],a=e.inputShapes[0],o=a[a.length-1],r=z(a.slice(0,-2)),l=i==="float16",s=y(i,!0),n=Math.min(256,I(0,t.limits.maxComputeWorkgroupSizeX)),p=o>=Oe,d=p?Le:o,m=`
${l?"enable f16;":""}
${O}

@group(0) @binding(0) var<storage, read> input: array<${s}>;
@group(0) @binding(1) var<storage, read_write> output: array<${s}>;

struct CholeskyParams {
  phase: u32,
  k: u32,
  block_size: u32,
  rows_below: u32,
}

@group(1) @binding(0) var<uniform> params: CholeskyParams;

// Shared memory for the diagonal element
var<workgroup> L_jj: ${s};

fn mat_idx(base: u32, row: u32, col: u32) -> u32 {
  return base + row * ${o}u + col;
}

@compute @workgroup_size(${n})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  let tid = local_id.x;

  if (params.phase == ${P.Unblocked}u) {
    let batch = wg_id.x + wg_id.y * ${B}u;
    if (batch >= ${r}u) {
      return;
    }

    let base = batch * ${o*o}u;

    // Zero out output and copy lower triangle from input.
    for (var idx = tid; idx < ${o*o}u; idx += ${n}u) {
      let row = idx / ${o}u;
      let col = idx % ${o}u;
      output[base + idx] = select(0, input[base + idx], col <= row);
    }
    storageBarrier();

    // Cholesky-Crout algorithm: process column by column.
    for (var j = 0u; j < ${o}u; j++) {
      for (var i = j + tid; i < ${o}u; i += ${n}u) {
        var sum = output[mat_idx(base, i, j)];
        for (var k = 0u; k < j; k++) {
          sum -= output[mat_idx(base, i, k)] * output[mat_idx(base, j, k)];
        }
        output[mat_idx(base, i, j)] = sum;
      }
      storageBarrier();

      if (tid == 0u) {
        L_jj = sqrt(output[mat_idx(base, j, j)]);
        output[mat_idx(base, j, j)] = L_jj;
      }
      workgroupBarrier();

      for (var i = j + 1u + tid; i < ${o}u; i += ${n}u) {
        output[mat_idx(base, i, j)] /= L_jj;
      }
      storageBarrier();
    }
    return;
  }

  if (params.phase == ${P.InitBlocked}u) {
    let batch = wg_id.x + wg_id.y * ${B}u;
    if (batch >= ${r}u) {
      return;
    }

    let base = batch * ${o*o}u;

    for (var idx = tid; idx < ${o*o}u; idx += ${n}u) {
      let row = idx / ${o}u;
      let col = idx % ${o}u;
      output[base + idx] = select(0, input[base + idx], col <= row);
    }
    return;
  }

  if (params.phase == ${P.FactorBlock}u) {
    let batch = wg_id.x + wg_id.y * ${B}u;
    if (batch >= ${r}u) {
      return;
    }

    let base = batch * ${o*o}u;
    let k0 = params.k;
    let b = params.block_size;

    for (var j = 0u; j < b; j++) {
      let col = k0 + j;
      for (var i = j + tid; i < b; i += ${n}u) {
        let row = k0 + i;
        var sum = output[mat_idx(base, row, col)];
        for (var r = 0u; r < j; r++) {
          sum -= output[mat_idx(base, row, k0 + r)] *
            output[mat_idx(base, col, k0 + r)];
        }
        output[mat_idx(base, row, col)] = sum;
      }
      storageBarrier();

      if (tid == 0u) {
        L_jj = sqrt(output[mat_idx(base, col, col)]);
        output[mat_idx(base, col, col)] = L_jj;
      }
      workgroupBarrier();

      for (var i = j + 1u + tid; i < b; i += ${n}u) {
        output[mat_idx(base, k0 + i, col)] /= L_jj;
      }
      storageBarrier();
    }
    return;
  }

  if (params.phase == ${P.SolvePanel}u) {
    let rows = params.rows_below;
    if (rows == 0u) {
      return;
    }

    let global = global_id.x + global_id.y * ${B*n}u;
    if (global >= rows * ${r}u) {
      return;
    }

    let batch = global / rows;
    let row = params.k + params.block_size + (global % rows);
    let base = batch * ${o*o}u;

    for (var j = 0u; j < params.block_size; j++) {
      let col = params.k + j;
      var sum = output[mat_idx(base, row, col)];
      for (var r = 0u; r < j; r++) {
        sum -= output[mat_idx(base, row, params.k + r)] *
          output[mat_idx(base, col, params.k + r)];
      }
      output[mat_idx(base, row, col)] =
        sum / output[mat_idx(base, col, col)];
    }
    return;
  }

  if (params.phase == ${P.UpdateTrailing}u) {
    let rows = params.rows_below;
    if (rows == 0u) {
      return;
    }

    let tile_count = (rows + ${L-1}u) / ${L}u;
    let tiles_per_batch = tile_count * (tile_count + 1u) / 2u;
    let wg_global = wg_id.x + wg_id.y * ${B}u;
    if (wg_global >= tiles_per_batch * ${r}u) {
      return;
    }

    let batch = wg_global / tiles_per_batch;
    let tile = wg_global % tiles_per_batch;
    var tile_row = u32((sqrt(f32(8u * tile + 1u)) - 1.0) * 0.5);
    // Correct for GPU sqrt rounding around exact triangular-number boundaries.
    var row_start = tile_row * (tile_row + 1u) / 2u;
    if (row_start > tile) {
      tile_row -= 1u;
      row_start = tile_row * (tile_row + 1u) / 2u;
    }
    let next_row_start = (tile_row + 1u) * (tile_row + 2u) / 2u;
    if (tile >= next_row_start) {
      tile_row += 1u;
      row_start = next_row_start;
    }
    let tile_col = tile - row_start;
    let local_row = tid / ${L}u;
    let local_col = tid % ${L}u;
    let row_local = tile_row * ${L}u + local_row;
    let col_local = tile_col * ${L}u + local_col;
    if (row_local >= rows || col_local >= rows) {
      return;
    }
    if (row_local < col_local) {
      return;
    }

    let row = params.k + params.block_size + row_local;
    let col = params.k + params.block_size + col_local;
    let base = batch * ${o*o}u;
    var sum = output[mat_idx(base, row, col)];
    for (var r = 0u; r < params.block_size; r++) {
      sum -= output[mat_idx(base, row, params.k + r)] *
        output[mat_idx(base, col, params.k + r)];
    }
    output[mat_idx(base, row, col)] = sum;
  }
}
`.trim(),w=[];if(!p)w.push({grid:U(r),uniform:G(P.Unblocked,0,o,0)});else{w.push({grid:U(r),uniform:G(P.InitBlocked,0,d,0)});for(let $=0;$<o;$+=d){const _=Math.min(d,o-$),g=o-$-_;w.push({grid:U(r),uniform:G(P.FactorBlock,$,_,g)}),g>0&&(w.push({grid:U(Math.ceil(g*r/n)),uniform:G(P.SolvePanel,$,_,g)}),w.push({grid:U(Math.ceil(g/L)*(Math.ceil(g/L)+1)*r/2),uniform:G(P.UpdateTrailing,$,_,g)}))}}return[{code:m,numInputs:1,numOutputs:1,hasUniform:!0,passes:w}]}function Ge(t,e){const i=e.inputDtypes[0],a=e.inputShapes[0],o=a[a.length-2],r=a[a.length-1],l=Math.min(o,r),s=z(a.slice(0,-2)),n=i==="float16",p=y(i,!0),d=I(Math.max(o,r),t.limits.maxComputeWorkgroupSizeX);return[{code:`
${n?"enable f16;":""}
${O}

@group(0) @binding(0) var<storage, read> input: array<${p}>;
@group(0) @binding(1) var<storage, read_write> lu: array<${p}>;
@group(0) @binding(2) var<storage, read_write> pivots: array<i32>;
@group(0) @binding(3) var<storage, read_write> perm: array<i32>;

var<workgroup> pivot_row: u32;
var<workgroup> pivot_val: ${p};

@compute @workgroup_size(${d})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${B}u;
  if (batch >= ${s}u) {
    return;
  }

  let lu_base = batch * ${o*r}u;
  let piv_base = batch * ${l}u;
  let perm_base = batch * ${o}u;
  let tid = local_id.x;

  // Copy input to lu
  for (var idx = tid; idx < ${o*r}u; idx += ${d}u) {
    lu[lu_base + idx] = input[lu_base + idx];
  }
  // Initialize permutation
  for (var idx = tid; idx < ${o}u; idx += ${d}u) {
    perm[perm_base + idx] = i32(idx);
  }
  storageBarrier();

  // LU decomposition with partial pivoting
  for (var j = 0u; j < ${l}u; j++) {
    // Step 1: Thread 0 finds pivot (max abs value in column j, rows >= j)
    if (tid == 0u) {
      var max_val = abs(lu[lu_base + j * ${r}u + j]);
      var max_row = j;
      for (var i = j + 1u; i < ${o}u; i++) {
        let val = abs(lu[lu_base + i * ${r}u + j]);
        if (val > max_val) {
          max_val = val;
          max_row = i;
        }
      }
      pivot_row = max_row;
      pivot_val = lu[lu_base + max_row * ${r}u + j];
      pivots[piv_base + j] = i32(max_row);
    }
    workgroupBarrier();

    // Step 2: Swap rows j and pivot_row (threads collaborate)
    let pr = pivot_row;
    if (pr != j) {
      for (var col = tid; col < ${r}u; col += ${d}u) {
        let tmp = lu[lu_base + j * ${r}u + col];
        lu[lu_base + j * ${r}u + col] = lu[lu_base + pr * ${r}u + col];
        lu[lu_base + pr * ${r}u + col] = tmp;
      }
      if (tid == 0u) {
        let tmp_p = perm[perm_base + j];
        perm[perm_base + j] = perm[perm_base + pr];
        perm[perm_base + pr] = tmp_p;
      }
    }
    storageBarrier();

    // Step 3: Compute L[i][j] and update submatrix
    // Each thread handles one row i > j
    for (var i = j + 1u + tid; i < ${o}u; i += ${d}u) {
      let factor = lu[lu_base + i * ${r}u + j] / pivot_val;
      lu[lu_base + i * ${r}u + j] = factor; // L[i][j]
      for (var k = j + 1u; k < ${r}u; k++) {
        lu[lu_base + i * ${r}u + k] -= factor * lu[lu_base + j * ${r}u + k];
      }
    }
    storageBarrier();
  }
}
`.trim(),numInputs:1,numOutputs:3,hasUniform:!1,passes:[{grid:U(s)}]}]}function Te(t,e,i){const a=e.inputDtypes[0],o=e.inputShapes[0],r=o[o.length-1],l=z(o.slice(0,-2)),s=a==="float16",n=y(a,!0),p=`${n}(${i.tolerance})`,d=I(Math.max(r,1),t.limits.maxComputeWorkgroupSizeX);return[{code:`
${s?"enable f16;":""}
${O}

@group(0) @binding(0) var<storage, read> input: array<${n}>;
@group(0) @binding(1) var<storage, read_write> diagonalized: array<${n}>;
@group(0) @binding(2) var<storage, read_write> vectors: array<${n}>;

var<workgroup> done: u32;
var<workgroup> rot_active: u32;
var<workgroup> rot_c: ${n};
var<workgroup> rot_s: ${n};
var<workgroup> rot_app: ${n};
var<workgroup> rot_aqq: ${n};
var<workgroup> rot_apq: ${n};

fn mat_idx(base: u32, row: u32, col: u32) -> u32 {
  return base + row * ${r}u + col;
}

fn sym_idx(base: u32, row: u32, col: u32) -> u32 {
  return mat_idx(base, max(row, col), min(row, col));
}

@compute @workgroup_size(${d})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${B}u;
  if (batch >= ${l}u) {
    return;
  }

  let base = batch * ${r*r}u;
  let tid = local_id.x;

  for (var idx = tid; idx < ${r*r}u; idx += ${d}u) {
    let row = idx / ${r}u;
    let col = idx % ${r}u;
    diagonalized[base + idx] = select(
      ${n}(0),
      input[base + idx],
      row >= col,
    );
    vectors[base + idx] = select(${n}(0), ${n}(1), row == col);
  }
  storageBarrier();

  for (var sweep = 0u; sweep < ${i.maxSweeps}u; sweep++) {
    if (tid == 0u) {
      var max_abs = ${n}(1);
      var max_offdiag = ${n}(0);
      for (var idx = 0u; idx < ${r*r}u; idx++) {
        let row = idx / ${r}u;
        let col = idx % ${r}u;
        let value = abs(diagonalized[base + idx]);
        max_abs = max(max_abs, value);
        if (row > col) {
          max_offdiag = max(max_offdiag, value);
        }
      }
      done = select(0u, 1u, max_offdiag <= ${p} * max_abs);
    }
    let done_uniform = workgroupUniformLoad(&done);
    if (done_uniform != 0u) {
      break;
    }

    for (var p = 0u; p + 1u < ${r}u; p++) {
      for (var q = p + 1u; q < ${r}u; q++) {
        if (tid == 0u) {
          rot_app = diagonalized[mat_idx(base, p, p)];
          rot_aqq = diagonalized[mat_idx(base, q, q)];
          rot_apq = diagonalized[sym_idx(base, p, q)];
          if (rot_apq == ${n}(0)) {
            rot_active = 0u;
            rot_c = ${n}(1);
            rot_s = ${n}(0);
          } else {
            let tau = (rot_aqq - rot_app) / (${n}(2) * rot_apq);
            let tau_sign = select(${n}(-1), ${n}(1), tau >= ${n}(0));
            let t = tau_sign / (abs(tau) + sqrt(tau * tau + ${n}(1)));
            rot_c = inverseSqrt(t * t + ${n}(1));
            rot_s = t * rot_c;
            rot_active = 1u;
          }
        }
        workgroupBarrier();

        if (rot_active != 0u) {
          for (var k = tid; k < ${r}u; k += ${d}u) {
            if (k != p && k != q) {
              let kp = sym_idx(base, k, p);
              let kq = sym_idx(base, k, q);
              let akp = diagonalized[kp];
              let akq = diagonalized[kq];
              let next_kp = rot_c * akp - rot_s * akq;
              let next_kq = rot_s * akp + rot_c * akq;
              diagonalized[kp] = next_kp;
              diagonalized[kq] = next_kq;
            } else if (k == p) {
              diagonalized[mat_idx(base, p, p)] =
                rot_c * rot_c * rot_app - ${n}(2) * rot_s * rot_c * rot_apq + rot_s * rot_s * rot_aqq;
              diagonalized[sym_idx(base, p, q)] = ${n}(0);
            } else {
              diagonalized[mat_idx(base, q, q)] =
                rot_s * rot_s * rot_app + ${n}(2) * rot_s * rot_c * rot_apq + rot_c * rot_c * rot_aqq;
            }

            let vp = mat_idx(base, k, p);
            let vq = mat_idx(base, k, q);
            let vkp = vectors[vp];
            let vkq = vectors[vq];
            vectors[vp] = rot_c * vkp - rot_s * vkq;
            vectors[vq] = rot_s * vkp + rot_c * vkq;
          }
        }
        storageBarrier();
      }
    }
  }
}
`.trim(),numInputs:1,numOutputs:2,hasUniform:!1,passes:[{grid:U(l)}]}]}function H(t,e,i,a){return new Uint8Array(new Uint32Array([t,e,i,a?1:0]).buffer)}function Fe(t,e,i){const a=e.inputDtypes[0],o=e.inputShapes[0],r=o[o.length-1],l=z(o.slice(0,-1));if(z(i.factors)!==r)throw new Error(`fft: factorization ${i.factors} does not match size ${r}`);const s=a==="float16",n=y(a,!0),p=Math.min(256,I(0,t.limits.maxComputeWorkgroupSizeX)),d=Math.max(1,...i.factors),m=i.inverse?"6.283185307179586":"-6.283185307179586",w=i.factors.map(h=>`
  digit = remaining % ${h}u;
  remaining = remaining / ${h}u;
  stride = stride * ${h}u;
  reversed = reversed + digit * (${r}u / stride);`).join(""),$=`
${s?"enable f16;":""}
${O}

@group(0) @binding(0) var<storage, read> input_real: array<${n}>;
@group(0) @binding(1) var<storage, read> input_imag: array<${n}>;
@group(0) @binding(2) var<storage, read_write> output_real: array<${n}>;
@group(0) @binding(3) var<storage, read_write> output_imag: array<${n}>;

struct FftParams {
  phase: u32,
  radix: u32,
  prev: u32,
  normalize: u32,
}

@group(1) @binding(0) var<uniform> fft_params: FftParams;

fn digit_reversed_index(index: u32) -> u32 {
  var remaining = index;
  var stride = 1u;
  var reversed = 0u;
  var digit = 0u;
${w}
  return reversed;
}

@compute @workgroup_size(${p})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global = global_id.x + global_id.y * ${B*p}u;

  if (fft_params.phase == 0u) {
    if (global >= ${l*r}u) {
      return;
    }
    let batch = global / ${r}u;
    let out_idx = global % ${r}u;
    let source = batch * ${r}u + digit_reversed_index(out_idx);
    output_real[global] = input_real[source];
    output_imag[global] = input_imag[source];
    return;
  }

  let butterflies_per_batch = ${r}u / fft_params.radix;
  if (global >= ${l}u * butterflies_per_batch) {
    return;
  }

  let batch = global / butterflies_per_batch;
  let local = global % butterflies_per_batch;
  let j = local % fft_params.prev;
  let group = local / fft_params.prev;
  let span = fft_params.prev * fft_params.radix;
  let start = batch * ${r}u + group * span + j;
  let scale = select(1.0, 1.0 / f32(${r}u), fft_params.normalize != 0u);

  var scratch_real: array<f32, ${d}>;
  var scratch_imag: array<f32, ${d}>;

  for (var q = 0u; q < fft_params.radix; q++) {
    let idx = start + q * fft_params.prev;
    let angle = ${m} * f32(q * j) / f32(span);
    let c = cos(angle);
    let s = sin(angle);
    let xr = f32(output_real[idx]);
    let xi = f32(output_imag[idx]);
    scratch_real[q] = xr * c - xi * s;
    scratch_imag[q] = xr * s + xi * c;
  }

  for (var p = 0u; p < fft_params.radix; p++) {
    var sum_real = 0.0;
    var sum_imag = 0.0;
    for (var q = 0u; q < fft_params.radix; q++) {
      let angle = ${m} * f32(q * p) / f32(fft_params.radix);
      let c = cos(angle);
      let s = sin(angle);
      let xr = scratch_real[q];
      let xi = scratch_imag[q];
      sum_real += xr * c - xi * s;
      sum_imag += xr * s + xi * c;
    }
    let idx = start + p * fft_params.prev;
    output_real[idx] = ${n}(sum_real * scale);
    output_imag[idx] = ${n}(sum_imag * scale);
  }
}
`.trim(),_=[{grid:U(Math.ceil(l*r/p)),uniform:H(0,1,1,!1)}];let g=1;for(let h=0;h<i.factors.length;h++){const f=i.factors[h];_.push({grid:U(Math.ceil(l*r/f/p)),uniform:H(1,f,g,i.inverse&&h===i.factors.length-1)}),g*=f}return[{code:$,numInputs:2,numOutputs:2,hasUniform:!0,passes:_}]}function Q(t,e){switch(e.name){case"Sort":return Me(t,e.type);case"Argsort":return Pe(t,e.type);case"Scatter":return Ie(t,e.type,e.params);case"TriangularSolve":return Ae(t,e.type,e.params);case"Cholesky":return Re(t,e.type);case"LU":return Ge(t,e.type);case"JacobiEigh":return Te(t,e.type,e.params);case"Fft":return Fe(t,e.type,e.params);default:throw new le(e.name,"webgpu")}}const X=4096,Z=new WeakMap;function We(t){return{querySet:t.createQuerySet({type:"timestamp",count:X}),resolve:t.createBuffer({size:X*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),dst:t.createBuffer({size:X*8,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),nextIndex:0,entries:[]}}function De(t,e,i,a,o){const r=de(i);r.properties.push(["passes",`${a}`]),r.properties.push(["source",o]),e.batch.entries.push({...r,beginIndex:e.beginIndex,endIndex:e.endIndex}),Ne(t)}function Ne(t){queueMicrotask(()=>{const e=Z.get(t);e&&e.entries.length>0&&(Xe(t,e),Z.set(t,We(t)))})}function Xe(t,e){if(e.entries.length===0)return;const i=e.nextIndex,a=t.createCommandEncoder();a.resolveQuerySet(e.querySet,0,i,e.resolve,0),a.copyBufferToBuffer(e.resolve,0,e.dst,0,i*8),t.queue.submit([a.finish()]);const{entries:o}=e;e.dst.mapAsync(GPUMapMode.READ).then(()=>{try{const r=new BigInt64Array(e.dst.getMappedRange()),l=r[o[o.length-1].endIndex],s=performance.now();for(const n of o)pe("webgpu",n,s+Number(r[n.beginIndex]-l)/1e6,s+Number(r[n.endIndex]-l)/1e6)}finally{e.dst.unmap(),e.querySet.destroy(),e.resolve.destroy(),e.dst.destroy()}})}const J=64*1024*1024,Ye=64;var et=class{device;type="webgpu";maxArgs;pipelines;syncReader;buffers;nextSlot;#e=new Map;#t;#i=new Map;constructor(t){this.device=t,this.maxArgs=this.device.limits.maxStorageBuffersPerShaderStage-1,this.pipelines=new Qe(t),this.syncReader=new Ee(t),this.buffers=new Map,this.nextSlot=1,this.#t=this.#o(4),t.addEventListener("uncapturederror",e=>{console.error("Uncaptured error in WebGPU backend:",e.error.message)})}malloc(t,e){if(e&&e.byteLength!==t)throw new Error("initialData size does not match buffer size");const i=Math.ceil(t/4)*4||4,a=t===0?this.#t:this.#s(i);if(e&&t>0)if(e.byteLength%4===0)this.device.queue.writeBuffer(a,0,e);else{const r=e.byteLength-e.byteLength%4;r>0&&this.device.queue.writeBuffer(a,0,e,0,r);const l=new Uint8Array(4);l.set(e.subarray(r)),this.device.queue.writeBuffer(a,r,l)}const o=this.nextSlot++;return this.buffers.set(o,{buffer:a,size:t,allocatedSize:i,ref:1}),o}incRef(t){const e=this.buffers.get(t);if(!e)throw new W(t);e.ref++}decRef(t){const e=this.buffers.get(t);if(!e)throw new W(t);e.ref--,e.ref===0&&(this.buffers.delete(t),e.buffer!==this.#t&&this.#n(e.buffer,e.allocatedSize))}async read(t,e,i){const{buffer:a,size:o}=this.#r(t);if(a===this.#t)return new Uint8Array;e===void 0&&(e=0),i===void 0&&(i=o-e);const r=Math.ceil(i/4)*4,l=this.#o(r,{read:!0});try{const s=this.device.createCommandEncoder();s.copyBufferToBuffer(a,e,l,0,r),this.device.queue.submit([s.finish()]),await ve(this.device,l);const n=l.getMappedRange();return new Uint8Array(n.slice(),0,i)}finally{l.destroy()}}readSync(t,e,i){const{buffer:a,size:o}=this.#r(t);return a===this.#t?new Uint8Array:(e===void 0&&(e=0),i===void 0&&(i=o-e),this.syncReader.read(a,e,i))}#a(t){const e=ne.hash(t);let i=this.#e.get(e);return i||(i=Ke(this.device,t),this.#e.set(e,i)),i}async prepareKernel(t){const e=this.#a(t),i=await this.pipelines.prepare(e);return new T(t,[{...e,pipeline:i}])}prepareKernelSync(t){const e=this.#a(t),i=this.pipelines.prepareSync(e);return new T(t,[{...e,pipeline:i}])}async prepareRoutine(t){const e=Q(this.device,t);return new T(t,await Promise.all(e.map(async i=>{const a=await this.pipelines.prepare(i);return{...i,pipeline:a}})))}prepareRoutineSync(t){return new T(t,Q(this.device,t).map(e=>{const i=this.pipelines.prepareSync(e);return{...e,pipeline:i}}))}dispatch(t,e,i){const a=e.map(r=>this.#r(r).buffer),o=i.map(r=>this.#r(r).buffer);Ve(this.device,t,a,o)}#r(t){const e=this.buffers.get(t);if(!e)throw new W(t);return{buffer:e.buffer,size:e.size}}#s(t){if(t>J)return this.#o(t);const e=this.#i.get(t),i=e?.pop();return e&&e.length===0&&this.#i.delete(t),i??this.#o(t)}#n(t,e){if(e>J){t.destroy();return}const i=this.#i.get(e);if(!i){this.#i.set(e,[t]);return}if(i.length>=Ye){t.destroy();return}i.push(t)}#o(t,{mapped:e=!1,read:i=!1}={}){if(i&&e)throw new Error("mapped and read cannot both be true");return this.device.createBuffer({size:t,usage:i?GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,mappedAtCreation:e})}};function Ke(t,e){const i=Be(t,e);if(i)return i;const a=ue(e),{nargs:o,reduction:r}=e,l=Array.from({length:o},(f,x)=>`in${x}`),s=new te;s.emitPreamble(t,[a.exp,a.epilogue]);const n=Array.from({length:o},()=>null);a.exp.fold(f=>{f.op==="GlobalIndex"&&(n[f.arg[0]]=f.dtype)}),a.epilogue?.fold(f=>{f.op==="GlobalIndex"&&(n[f.arg[0]]=f.dtype)});for(let f=0;f<o;f++){const x=y(n[f]??"float32",!0);s.emit(`@group(0) @binding(${f}) var<storage, read> ${l[f]} : array<${x}>;`)}const p=y(e.dtype,!0);s.emit(`@group(0) @binding(${o}) var<storage, read_write> result : array<${p}>;`);const d=r?a.size.groups??1:1,m=r&&d>1;if(m&&a.threadCount%d!==0)throw new Error("WebGPU grouped reduction has invalid thread count");if(m&&d>t.limits.maxComputeWorkgroupSizeX)throw new Error("WebGPU grouped reduction exceeds workgroup size limit");const w=m?d:I(a.threadCount,256),$=m?a.threadCount/d:Math.ceil(a.threadCount/w),[_,g]=U($);if(m){const f=y(r.dtype);for(let x=0;x<(a.size.upcast??1);x++)s.emit(`var<workgroup> partial${x}: array<${f}, ${d}>;`)}if(s.emit("",`@compute @workgroup_size(${w})`),m)s.emit("fn main(",s.pushIndent,"@builtin(local_invocation_id) lid : vec3<u32>,","@builtin(workgroup_id) wg_id : vec3<u32>,",s.popIndent,") {",s.pushIndent),g===1?s.emit(`if (wg_id.x >= ${$}u) { return; }`,"let gidx: i32 = i32(wg_id.x);"):s.emit(`if (${_}u * wg_id.y + wg_id.x >= ${$}u) { return; }`,`let gidx: i32 = i32(${_}u * wg_id.y + wg_id.x);`),s.emit("let group: i32 = i32(lid.x);");else if(s.emit("fn main(@builtin(global_invocation_id) id : vec3<u32>) {",s.pushIndent),g===1)s.emit(`if (id.x >= ${a.threadCount}) { return; }`,"let gidx: i32 = i32(id.x);");else{const f=_*w;s.emit(`if (${f} * id.y + id.x >= ${a.threadCount}) { return; }`,`let gidx: i32 = i32(${f} * id.y + id.x);`)}s.emitPhonyAssignments(l);const h=new ae(s,l);if(r){const f=a.size.unroll??1,x=a.size.upcast??1,v=[...Array(x)].map((u,c)=>`acc${c}`);for(let u=0;u<x;u++)s.emit(`var ${v[u]}: ${y(r.dtype)} = ${ie(r.dtype,r.identity)};`);s.emit(`for (var ridx: i32 = 0; ridx < ${a.size.reduce}; ridx++) {`,s.pushIndent);const M=[],A=new Map;for(let u=0;u<x;u++){M.push([]);for(let c=0;c<f;c++){const k=a.exp.substitute({upcast:q.i32(u),unroll:q.i32(c)});M[u].push(k.simplify(A)),h.countReferences(M[u][c])}}const E=M.map(u=>u.map(c=>h.run(c)).map(b));for(let u=0;u<x;u++){let c=E[u][0];for(let k=1;k<f;k++)if(r.op==="Add")c=`${c} + ${E[u][k]}`;else if(r.op==="Mul")c=`${c} * ${E[u][k]}`;else if(r.op==="Min")c=r.dtype==="bool"?`(${c} && ${E[u][k]})`:`min(${c}, ${E[u][k]})`;else if(r.op==="Max")c=r.dtype==="bool"?`(${c} || ${E[u][k]})`:`max(${c}, ${E[u][k]})`;else throw new Error(`Unsupported reduction op: ${r.op}`);if(r.op==="Add")s.emit(`${v[u]} += ${c};`);else if(r.op==="Mul")s.emit(`${v[u]} *= ${c};`);else if(r.op==="Min")r.dtype==="bool"?s.emit(`${v[u]} = ${v[u]} && ${c};`):s.emit(`${v[u]} = min(${v[u]}, ${c});`);else if(r.op==="Max")r.dtype==="bool"?s.emit(`${v[u]} = ${v[u]} || ${c};`):s.emit(`${v[u]} = max(${v[u]}, ${c});`);else throw new Error(`Unsupported reduction op: ${r.op}`)}if(s.emit(s.popIndent,"}"),m){for(let u=0;u<x;u++)s.emit(`partial${u}[lid.x] = ${v[u]};`);s.emit("workgroupBarrier();");for(let u=d/2;u>=1;u/=2){s.emit(`if (lid.x < ${u}u) {`,s.pushIndent);for(let c=0;c<x;c++)s.emit(`partial${c}[lid.x] = ${_e(r.op,r.dtype,`partial${c}[lid.x]`,`partial${c}[lid.x + ${u}u]`)};`);s.emit(s.popIndent,"}","workgroupBarrier();")}}h.reset();const C=[],j=[];for(let u=0;u<x;u++){const c=a.outputIdxExp.substitute({upcast:q.i32(u)});C.push(c.simplify(A)),h.countReferences(C[u]),j.push(a.epilogue.substitute({acc:q.variable(r.dtype,v[u]),upcast:q.i32(u)}).simplify(A)),h.countReferences(j[u])}if(m){s.emit("if (lid.x == 0u) {",s.pushIndent);for(let u=0;u<x;u++)s.emit(`${v[u]} = partial${u}[0u];`)}for(let u=0;u<x;u++){const c=b(h.run(C[u]));let k=b(h.run(j[u]));p!==y(j[u].dtype)&&(k=`${p}(${k})`),s.emit(`result[${c}] = ${k};`)}m&&s.emit(s.popIndent,"}")}else{h.countReferences(a.exp);let f=b(h.run(a.exp));p!==y(a.exp.dtype)&&(f=`${p}(${f})`),s.emit(`result[gidx] = ${f};`)}return s.emit(s.popIndent,"}"),{code:s.toString(),numInputs:o,numOutputs:1,hasUniform:!1,passes:[{grid:[_,g]}]}}function Ve(t,e,i,a){const{data:o,source:r}=e,l=t.createCommandEncoder();for(const{pipeline:s,...n}of o){if(i.length!==n.numInputs||a.length!==n.numOutputs)throw new Error(`webgpu: expected ${n.numInputs} inputs and ${n.numOutputs} outputs, got ${i.length} inputs and ${a.length} outputs`);if(n.clearOutputs)for(const _ of a)l.clearBuffer(_);const p=n.passes.filter(({grid:_})=>z(_)>0);if(p.length===0)continue;const d=void 0,m=t.createBindGroup({layout:s.getBindGroupLayout(0),entries:[...i.map((_,g)=>({binding:g,resource:{buffer:_}})),...a.map((_,g)=>({binding:i.length+g,resource:{buffer:_}}))]});let w=null,$=0;if(n.hasUniform){const[_,g]=He(t,p.map(({uniform:h})=>h));$=g,w=t.createBindGroup({layout:s.getBindGroupLayout(1),entries:[{binding:0,resource:{buffer:_,size:g}}]})}for(let _=0;_<p.length;_++){const{grid:g}=p[_];let h;if(d){const x=_===0,v=_===p.length-1;(x||v)&&(h={querySet:d.batch.querySet,...x?{beginningOfPassWriteIndex:d.beginIndex}:{},...v?{endOfPassWriteIndex:d.endIndex}:{}})}const f=l.beginComputePass({timestampWrites:h});f.setPipeline(s),f.setBindGroup(0,m),w&&f.setBindGroup(1,w,[_*$]),f.dispatchWorkgroups(g[0],g[1]),f.end()}d&&De(t,d,r,p.length,n.code)}t.queue.submit([l.finish()])}function He(t,e){for(const l of e)if(!l||l.byteLength===0||l.byteLength!==e[0].byteLength)throw new Error("webgpu: Uniform mismatch between shader passes");const i=t.limits.minUniformBufferOffsetAlignment,a=Math.ceil(e[0].byteLength/i)*i,o=t.createBuffer({size:a*e.length,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:!0}),r=new Uint8Array(o.getMappedRange());for(let l=0;l<e.length;l++)r.set(e[l],l*a);return o.unmap(),[o,a]}var Qe=class{device;cache;inProgress;constructor(t){this.device=t,this.cache=new Map,this.inProgress=new Map}#e(t){if(t.numInputs+t.numOutputs>this.device.limits.maxStorageBuffersPerShaderStage){const i=t.numInputs+t.numOutputs,a=this.device.limits.maxStorageBuffersPerShaderStage;throw new Error(`Too many buffers (${i}) for WebGPU pipeline (max: ${a})`)}const e=[this.device.createBindGroupLayout({entries:F(t.numInputs+t.numOutputs).map(i=>({binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:i<t.numInputs?"read-only-storage":"storage"}}))})];return t.hasUniform&&e.push(this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:!0}}]})),this.device.createPipelineLayout({bindGroupLayouts:e})}async prepare(t){const e=this.cache.get(t.code);if(e)return e;const i=this.inProgress.get(t.code);if(i)return await i;const a=this.device.createShaderModule({code:t.code}),o=(async()=>{this.device.pushErrorScope("validation");try{const l=await this.device.createComputePipelineAsync({layout:this.#e(t),compute:{module:a,entryPoint:"main"}});return await this.device.popErrorScope(),l}catch{const s=await ee(a,await this.device.popErrorScope(),t.code);throw new Error(s)}})();this.inProgress.set(t.code,o);const r=await o;return this.cache.set(t.code,r),r}prepareSync(t){const e=this.cache.get(t.code);if(e)return e;const i=this.device.createShaderModule({code:t.code});this.device.pushErrorScope("validation");const a=this.device.createComputePipeline({layout:this.#e(t),compute:{module:i,entryPoint:"main"}});return this.device.popErrorScope().then(async o=>{if(o!==null){const r=await ee(i,o,t.code);console.error(r)}}),this.cache.set(t.code,a),a}};async function ee(t,e,i){let a=`Failed to compile shader: ${e?e.message:"(no error scope)"}`;const o=await t.getCompilationInfo();for(const r of o.messages)a+=`
  [${r.type} at ${r.lineNum}:${r.linePos}] ${r.message}`;return i&&(a+=`

${i}`),a}export{et as WebGPUBackend};
