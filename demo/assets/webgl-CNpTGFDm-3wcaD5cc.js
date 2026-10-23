import{S as m,E as g,U,d as S,s as h,r as B,i as b,a as R,b as v}from"./jaxjs-Dm1kCLlx.js";import"./index-B6m_Gp3N.js";const M=`
uvec2 threefry2x32(uvec2 key, uvec2 ctr) {
  uint ks0 = key.x;
  uint ks1 = key.y;
  uint ks2 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  uint x0 = ctr.x + ks0;
  uint x1 = ctr.y + ks1;

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

  return uvec2(x0, x1);
}`,D=`
const float _erf_p = 0.3275911;
const float _erf_a1 = 0.254829592;
const float _erf_a2 = -0.284496736;
const float _erf_a3 = 1.421413741;
const float _erf_a4 = -1.453152027;
const float _erf_a5 = 1.061405429;
float erf(float x) {
  float t = 1.0 / (1.0 + _erf_p * abs(x));
  float P_t = (((((_erf_a5 * t) + _erf_a4) * t + _erf_a3) * t + _erf_a2) * t + _erf_a1) * t;
  return sign(x) * (1.0 - P_t * exp(-x * x));
}
float erfc(float x) {
  float t = 1.0 / (1.0 + _erf_p * abs(x));
  float P_t = (((((_erf_a5 * t) + _erf_a4) * t + _erf_a3) * t + _erf_a2) * t + _erf_a1) * t;
  float E = P_t * exp(-x * x);
  return x >= 0.0 ? E : 2.0 - E;
}`,N=`
int bitCountFallback(uint value) {
  value = value - ((value >> 1u) & 0x55555555u);
  value = (value & 0x33333333u) + ((value >> 2u) & 0x33333333u);
  value = (value + (value >> 4u)) & 0x0f0f0f0fu;
  value = value + (value >> 8u);
  value = value + (value >> 16u);
  return int(value & 0x3fu);
}
int bitCountFallback(int value) {
  return bitCountFallback(uint(value));
}`;var V=class{type="webgl";maxArgs=8;gl;#t;#e;#r;#i;constructor(t){this.gl=t,this.#t=t.createFramebuffer(),this.#e=new Map,this.#r=new Map,this.#i=1}malloc(t,i){const n=this.gl,e=Math.ceil(t/4)||1,{width:r,height:d}=y(Math.ceil(e/4)||1),a=n.createTexture();if(!a)throw new Error("Failed to create texture");n.bindTexture(n.TEXTURE_2D,a),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_MIN_FILTER,n.NEAREST),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_MAG_FILTER,n.NEAREST),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_WRAP_S,n.CLAMP_TO_EDGE),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_WRAP_T,n.CLAMP_TO_EDGE);const x=r*d*4;let l=null;i&&(l=new Uint32Array(x),new Uint8Array(l.buffer).set(i)),n.texImage2D(n.TEXTURE_2D,0,n.RGBA32UI,r,d,0,n.RGBA_INTEGER,n.UNSIGNED_INT,l),n.bindTexture(n.TEXTURE_2D,null);const u=this.#i++;return this.#e.set(u,{ref:1,size:t,texture:a,width:r,height:d}),u}incRef(t){const i=this.#e.get(t);if(!i)throw new m(t);i.ref++}decRef(t){const i=this.#e.get(t);if(!i)throw new m(t);i.ref--,i.ref===0&&(this.gl.deleteTexture(i.texture),this.#e.delete(t))}async read(t,i,n){const e=this.#e.get(t);if(!e)throw new m(t);const r=this.gl;i===void 0&&(i=0),n===void 0&&(n=e.size-i),r.bindFramebuffer(r.FRAMEBUFFER,this.#t),r.framebufferTexture2D(r.FRAMEBUFFER,r.COLOR_ATTACHMENT0,r.TEXTURE_2D,e.texture,0);const d=e.width*e.height*4*4,a=new Uint32Array(d/4),x=r.createBuffer();if(!x)throw new Error("Failed to create PBO");r.bindBuffer(r.PIXEL_PACK_BUFFER,x),r.bufferData(r.PIXEL_PACK_BUFFER,d,r.STREAM_READ),r.readPixels(0,0,e.width,e.height,r.RGBA_INTEGER,r.UNSIGNED_INT,0);const l=r.getError();if(l!==r.NO_ERROR)throw r.deleteBuffer(x),new Error(`WebGL error after readPixels: ${l}`);const u=r.fenceSync(r.SYNC_GPU_COMMANDS_COMPLETE,0);if(!u)throw new Error("Failed to create sync object");r.flush(),r.bindBuffer(r.PIXEL_PACK_BUFFER,null),r.bindFramebuffer(r.FRAMEBUFFER,null),await new Promise((s,f)=>{const c=()=>{const $=r.clientWaitSync(u,0,0);if($===r.TIMEOUT_EXPIRED){setTimeout(c,5);return}if($===r.WAIT_FAILED){r.deleteSync(u),r.deleteBuffer(x),f(new Error("clientWaitSync failed"));return}s()};c()}),r.deleteSync(u),r.bindBuffer(r.PIXEL_PACK_BUFFER,x),r.getBufferSubData(r.PIXEL_PACK_BUFFER,0,a),r.bindBuffer(r.PIXEL_PACK_BUFFER,null),r.deleteBuffer(x);const o=new Uint8Array(a.buffer);return new Uint8Array(o.slice(i,i+n))}readSync(t,i,n){const e=this.#e.get(t);if(!e)throw new m(t);const r=this.gl;i===void 0&&(i=0),n===void 0&&(n=e.size-i),r.bindFramebuffer(r.FRAMEBUFFER,this.#t),r.framebufferTexture2D(r.FRAMEBUFFER,r.COLOR_ATTACHMENT0,r.TEXTURE_2D,e.texture,0);const d=e.width*e.height*4,a=new Uint32Array(d);r.readPixels(0,0,e.width,e.height,r.RGBA_INTEGER,r.UNSIGNED_INT,a),r.bindFramebuffer(r.FRAMEBUFFER,null);const x=new Uint8Array(a.buffer);return new Uint8Array(x.slice(i,i+n))}async prepareKernel(t){return this.prepareKernelSync(t)}prepareKernelSync(t){const i=L(t),n=this.#r.get(i.code);if(n)return new g(t,n);const e=O(this.gl,i);return this.#r.set(i.code,e),new g(t,e)}prepareRoutine(t){throw new U(t.name,"webgl")}prepareRoutineSync(t){throw new U(t.name,"webgl")}dispatch(t,i,n){const e=this.gl;if(e.isContextLost())throw new Error("WebGL context lost - cannot dispatch");const{program:r,inputLocations:d}=t.data;if(i.length!==t.data.numInputs)throw new Error(`Expected ${t.data.numInputs} inputs, got ${i.length}`);if(n.length!==1)throw new Error(`Expected 1 output, got ${n.length}`);const a=this.#e.get(n[0]);if(!a)throw new m(n[0]);e.bindFramebuffer(e.FRAMEBUFFER,this.#t),e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,a.texture,0);const x=e.checkFramebufferStatus(e.FRAMEBUFFER);if(x!==e.FRAMEBUFFER_COMPLETE)throw new Error(`Framebuffer incomplete: ${x}`);e.viewport(0,0,a.width,a.height),e.useProgram(r);for(let u=0;u<i.length;u++){const o=this.#e.get(i[u]);if(!o)throw new m(i[u]);e.activeTexture(e.TEXTURE0+u),e.bindTexture(e.TEXTURE_2D,o.texture),d[u]!==null&&e.uniform1i(d[u],u)}e.drawArrays(e.TRIANGLES,0,3);const l=e.getError();if(l!==e.NO_ERROR){let u;throw l===e.INVALID_ENUM?u="INVALID_ENUM":l===e.INVALID_VALUE?u="INVALID_VALUE":l===e.INVALID_OPERATION?u="INVALID_OPERATION":l===e.INVALID_FRAMEBUFFER_OPERATION?u="INVALID_FRAMEBUFFER_OPERATION":l===e.OUT_OF_MEMORY?u="OUT_OF_MEMORY":l===e.CONTEXT_LOST_WEBGL?u="CONTEXT_LOST_WEBGL":u=`UNKNOWN(${l})`,new Error(`WebGL error after drawArrays: ${u}`)}e.bindFramebuffer(e.FRAMEBUFFER,null),e.useProgram(null)}};function L(t){const i=S(t),{nargs:n,reduction:e}=t,r=t.dtype,d=y(Math.ceil(t.size/4)||1),a=Array(n).fill("float32"),x={erf:!1,bitCount:!1,threefry:!1},l=E=>{E.op==="GlobalIndex"?a[E.arg[0]]=E.dtype:E.op==="Erf"||E.op==="Erfc"?x.erf=!0:E.op==="BitCount"?x.bitCount=!0:E.op==="Threefry2x32"&&(x.threefry=!0)};i.exp.fold(l),i.epilogue?.fold(l);const u=[];let o="";const s=Symbol("pushIndent"),f=Symbol("popIndent"),c=(...E)=>{for(const T of E)T===s?o+="  ":T===f?o=o.slice(0,-2):u.push(T&&o+T)};c("#version 300 es","precision highp float;","precision highp int;","");const $=Array.from({length:n},(E,T)=>`in${T}`),p=w(r);for(let E=0;E<n;E++)c(`uniform highp usampler2D ${$[E]};`);c("out uvec4 out0;");const _=new Set;for(const E of a)_.add(E);for(const E of _)c(G(E));if(x.erf&&c(D),x.bitCount&&c(N),x.threefry&&c(M),c(`${p} compute(int gidx) {`,s,`${p} result = ${A(r,0)};`,`if (gidx < ${t.size}) {`,s),!e)c(`result = ${h(F(i.exp,$,a))};`);else{c(`${w(e.dtype)} acc = ${A(e.dtype,e.identity)};`,`for (int ridx = 0; ridx < ${i.size.reduce}; ridx++) {`,s);const E=F(i.exp,$,a);if(e.op==="Add")c(`acc += ${h(E)};`);else if(e.op==="Mul")c(`acc *= ${h(E)};`);else if(e.op==="Min")e.dtype!=="bool"?c(`acc = min(acc, ${h(E)});`):c(`acc = acc && ${E};`);else if(e.op==="Max")e.dtype!=="bool"?c(`acc = max(acc, ${h(E)});`):c(`acc = acc || ${E};`);else throw new Error(`Unsupported reduction op: ${e.op}`);c(f,"}"),c(`result = ${F(i.epilogue,$,a)};`)}return c(f,"}","return result;",f,`}
`),c("void main() {",s,"ivec2 fragCoord = ivec2(gl_FragCoord.xy);",`int texelIdx = fragCoord.y * ${d.width} + fragCoord.x;`,`${p} result0 = compute(texelIdx * 4);`,`${p} result1 = compute(texelIdx * 4 + 1);`,`${p} result2 = compute(texelIdx * 4 + 2);`,`${p} result3 = compute(texelIdx * 4 + 3);`,`out0 = uvec4(${B(4).map(E=>k(r,`result${E}`)).join(", ")});`),c(f,"}"),{code:u.join(`
`),numInputs:n,outputSize:[d.width,d.height],outputDtype:r}}function I(t,i,n){const e=t.createShader(i);if(t.shaderSource(e,n),t.compileShader(e),!t.getShaderParameter(e,t.COMPILE_STATUS))throw new Error(t.getShaderInfoLog(e)??"Unknown shader compile error");return e}function P(t,i,n){const e=t.createProgram();if(t.attachShader(e,I(t,t.VERTEX_SHADER,i)),t.attachShader(e,I(t,t.FRAGMENT_SHADER,n)),t.linkProgram(e),!t.getProgramParameter(e,t.LINK_STATUS))throw new Error(t.getProgramInfoLog(e)??"Unknown program link error");return e}const C=`#version 300 es
precision highp float;
const vec2 pos[3] = vec2[](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
void main() { gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0); }
`;function O(t,i){const n=P(t,C,i.code),e=[];for(let r=0;r<i.numInputs;r++)e.push(t.getUniformLocation(n,`in${r}`));return{...i,program:n,inputLocations:e}}function y(t){let n=Math.min(Math.ceil(Math.sqrt(t)),16384);n=Math.min(1<<Math.ceil(Math.log2(n)),16384);const e=Math.min(Math.ceil(t/n),16384);return{width:n,height:e}}function w(t){switch(t){case"float32":return"float";case"int32":return"int";case"uint32":return"uint";case"bool":return"bool";default:throw new Error(`Unsupported dtype for WebGL: ${t}`)}}function G(t){const i=`load_${t}`,n=w(t);let e;if(b(t))e="uintBitsToFloat(val)";else if(t==="int32")e="int(val)";else if(t==="uint32")e="val";else if(t==="bool")e="val != 0u";else throw new Error(`Unsupported dtype for WebGL fetch: ${t}`);return`
${n} ${i}(highp usampler2D tex, int idx) {
  ivec2 texSize = textureSize(tex, 0);
  int texel = idx / 4;
  int component = idx - texel * 4;
  ivec2 coord = ivec2(texel % texSize.x, texel / texSize.x);
  uvec4 texVal = texelFetch(tex, coord, 0);
  uint val;
  if (component == 0) val = texVal.x;
  else if (component == 1) val = texVal.y;
  else if (component == 2) val = texVal.z;
  else val = texVal.w;
  return ${e};
}
`}function k(t,i){switch(t){case"float32":return`floatBitsToUint(${i})`;case"int32":return`uint(${i})`;case"uint32":return i;case"bool":return`(${i} ? 1u : 0u)`;default:throw new Error(`Unsupported dtype for WebGL output: ${t}`)}}function A(t,i){switch(t){case"bool":return i?"true":"false";case"int32":return i.toString();case"uint32":return i.toString()+"u";case"float32":return Number.isNaN(i)?"uintBitsToFloat(0x7fc00000u)":Number.isFinite(i)?"float("+i.toString()+")":i>0?"uintBitsToFloat(0x7f800000u)":"uintBitsToFloat(0xff800000u)";default:throw new Error(`Unsupported dtype for WebGL constant: ${t}`)}}function F(t,i,n){const e=new Map,r=d=>{if(e.has(d))return e.get(d);const{op:a,src:x,dtype:l,arg:u}=d;let o="";if(R.Binary.has(a)){const s=r(x[0]),f=r(x[1]);if(a==="Add")l==="bool"?o=`(${s} || ${f})`:o=`(${s} + ${f})`;else if(a==="Sub")o=`(${s} - ${f})`;else if(a==="Mul")l==="bool"?o=`(${s} && ${f})`:o=`(${s} * ${f})`;else if(a==="Idiv")b(l)?o=`trunc(${s} / ${f})`:o=`(${s} / ${f})`;else if(a==="Mod")b(l)?o=`(${s} - ${f} * trunc(${s} / ${f}))`:o=`(${s} % ${f})`;else if(a==="Min")l==="bool"?o=`(${s} && ${f})`:o=`min(${s}, ${f})`;else if(a==="Max")l==="bool"?o=`(${s} || ${f})`:o=`max(${s}, ${f})`;else if(a==="BitCombine"){let c=u==="and"?"&":u==="or"?"|":"^";l==="bool"&&(c=c+c),o=`(${s} ${c} ${f})`}else a==="BitShift"&&(u==="shl"?o=`(${s} << ${f})`:o=`(${s} >> ${f})`)}else if(R.Compare.has(a)){const s=r(x[0]),f=r(x[1]);a==="Cmplt"?o=`(${s} < ${f})`:a==="Cmpne"&&(b(x[0].dtype)?o=`(${s} != ${f} || isnan(${s}) || isnan(${f}))`:o=`(${s} != ${f})`)}else if(R.Unary.has(a)){const s=r(x[0]);if(a==="Sin")o=`sin(${h(s)})`;else if(a==="Cos")o=`cos(${h(s)})`;else if(a==="Asin")o=`asin(${h(s)})`;else if(a==="Atan")o=`atan(${h(s)})`;else if(a==="Exp")o=`exp(${h(s)})`;else if(a==="Log")o=`log(${h(s)})`;else if(a==="Erf")o=`erf(${h(s)})`;else if(a==="Erfc")o=`erfc(${h(s)})`;else if(a==="Sqrt")o=`sqrt(${h(s)})`;else if(a==="Floor")o=`floor(${h(s)})`;else if(a==="Ceil")o=`ceil(${h(s)})`;else if(a==="Reciprocal")o=`(1.0 / ${s})`;else if(a==="Cast")o=`${w(l)}(${h(s)})`;else if(a==="Bitcast"){const f=x[0].dtype;l===f?o=s:l==="float32"?f==="int32"?o=`intBitsToFloat(${h(s)})`:f==="uint32"&&(o=`uintBitsToFloat(${h(s)})`):l==="int32"?f==="float32"?o=`floatBitsToInt(${h(s)})`:f==="uint32"&&(o=`int(${h(s)})`):l==="uint32"&&(f==="float32"?o=`floatBitsToUint(${h(s)})`:f==="int32"&&(o=`uint(${h(s)})`))}else a==="BitCount"&&(o=`bitCountFallback(${h(s)})`)}else if(a==="Threefry2x32"){const[s,f,c,$]=x.map(E=>h(r(E))),p=u,_=`threefry2x32(uvec2(${s}, ${f}), uvec2(${c}, ${$}))`;p==="xor"?o=`(${_}.x ^ ${_}.y)`:p===0?o=`${_}.x`:p===1&&(o=`${_}.y`)}else if(a==="Where"){const[s,f,c]=x.map(r);o=`(${s} ? ${f} : ${c})`}else if(a==="Const")o=A(l,u);else if(a==="Special")o=u[0];else if(a==="Variable")o=u;else if(a==="GlobalIndex"){const s=u[0],f=r(x[0]);o=`load_${n[s]}(${i[s]}, ${h(f)})`}if(!o)throw new v(a,l,"webgl",u);return e.set(d,o),o};return r(t)}export{V as WebGLBackend};
