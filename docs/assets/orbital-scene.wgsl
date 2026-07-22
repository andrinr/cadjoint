// Generated from jaxcad.viewer.playground.EXAMPLE_SOURCE.
// Regenerate with compile_source(EXAMPLE_SOURCE)["shader"].

fn _where(_arg0: bool, _arg1: f32, _arg2: f32) -> f32 {
    let _v0: f32 = select(_arg2, _arg1, _arg0);
    return _v0;
}

fn norm_8(_arg0: vec2<f32>) -> f32 {
    let _v0: f32 = 0.000000;
    let _v1: vec2<f32> = _arg0 * _arg0;
    let _v2: f32 = _v0 + dot(_v1, vec2<f32>(1.0, 1.0));
    let _v3: f32 = sqrt(_v2);
    return _v3;
}

fn norm_0(_arg0: vec3<f32>) -> f32 {
    let _v0: f32 = 0.000000;
    let _v1: vec3<f32> = _arg0 * _arg0;
    let _v2: f32 = _v0 + dot(_v1, vec3<f32>(1.0, 1.0, 1.0));
    let _v3: f32 = sqrt(_v2);
    return _v3;
}

fn norm(_arg0: vec3<f32>) -> f32 {
    let _v0: f32 = 0.000000;
    let _v1: vec3<f32> = _arg0 * _arg0;
    let _v2: f32 = _v0 + dot(_v1, vec3<f32>(1.0, 1.0, 1.0));
    let _v3: f32 = sqrt(_v2);
    return _v3;
}

fn sdf(p: vec3<f32>) -> f32 {
    let _v0: f32 = -3.402823e38;
    let _v1: f32 = 0.000000;
    let _v2: f32 = 0.250000;
    let _v3: f32 = 0.000000;
    let _v4: f32 = 0.000000;
    let _v5: f32 = 4.000000;
    let _v6: f32 = 1.000000;
    let _v7: f32 = 0.620000;
    let _v8: vec3<f32> = vec3<f32>(1.000000, 0.000000, 0.000000);
    let _v9: f32 = 1.256637;
    let _v10: f32 = 0.950000;
    let _v11: f32 = 0.180000;
    let _v12: f32 = 0.100000;
    let _v13: vec3<f32> = vec3<f32>(0.000000, 0.000000, 1.000000);
    let _v14: f32 = 0.785398;
    let _v15: vec3<f32> = vec3<f32>(0.300000, 1.500000, 0.300000);
    let _v16: f32 = 0.040000;
    let _v17: vec3<f32> = vec3<f32>(1.180000, 0.220000, 0.080000);
    let _v18: f32 = 0.280000;
    let _v19: f32 = 0.060000;
    let _v20: f32 = norm(p);
    let _v21: f32 = f32(_v7);
    let _v22: f32 = _v20 - _v21;
    let _v23: f32 = norm_0(_v8);
    let _v24: vec3<f32> = vec3<f32>(_v23);
    let _v25: vec3<f32> = _v8 / _v24;
    let _v26: f32 = cos(_v9);
    let _v27: f32 = sin(_v9);
    let _v28: f32 = _v6 - _v26;
    let _v29: f32 = _v25.x;
    let _v30: f32 = _v29;
    let _v31: f32 = _v25.y;
    let _v32: f32 = _v31;
    let _v33: f32 = _v25.z;
    let _v34: f32 = _v33;
    let _v35: f32 = f32(_v28);
    let _v36: f32 = _v35 * _v30;
    let _v37: f32 = _v36 * _v30;
    let _v38: f32 = f32(_v26);
    let _v39: f32 = _v37 + _v38;
    let _v40: f32 = f32(_v28);
    let _v41: f32 = _v40 * _v30;
    let _v42: f32 = _v41 * _v32;
    let _v43: f32 = f32(_v27);
    let _v44: f32 = _v43 * _v34;
    let _v45: f32 = _v42 - _v44;
    let _v46: f32 = f32(_v28);
    let _v47: f32 = _v46 * _v30;
    let _v48: f32 = _v47 * _v34;
    let _v49: f32 = f32(_v27);
    let _v50: f32 = _v49 * _v32;
    let _v51: f32 = _v48 + _v50;
    let _v52: f32 = f32(_v28);
    let _v53: f32 = _v52 * _v30;
    let _v54: f32 = _v53 * _v32;
    let _v55: f32 = f32(_v27);
    let _v56: f32 = _v55 * _v34;
    let _v57: f32 = _v54 + _v56;
    let _v58: f32 = f32(_v28);
    let _v59: f32 = _v58 * _v32;
    let _v60: f32 = _v59 * _v32;
    let _v61: f32 = f32(_v26);
    let _v62: f32 = _v60 + _v61;
    let _v63: f32 = f32(_v28);
    let _v64: f32 = _v63 * _v32;
    let _v65: f32 = _v64 * _v34;
    let _v66: f32 = f32(_v27);
    let _v67: f32 = _v66 * _v30;
    let _v68: f32 = _v65 - _v67;
    let _v69: f32 = f32(_v28);
    let _v70: f32 = _v69 * _v30;
    let _v71: f32 = _v70 * _v34;
    let _v72: f32 = f32(_v27);
    let _v73: f32 = _v72 * _v32;
    let _v74: f32 = _v71 - _v73;
    let _v75: f32 = f32(_v28);
    let _v76: f32 = _v75 * _v32;
    let _v77: f32 = _v76 * _v34;
    let _v78: f32 = f32(_v27);
    let _v79: f32 = _v78 * _v30;
    let _v80: f32 = _v77 + _v79;
    let _v81: f32 = f32(_v28);
    let _v82: f32 = _v81 * _v34;
    let _v83: f32 = _v82 * _v34;
    let _v84: f32 = f32(_v26);
    let _v85: f32 = _v83 + _v84;
    let _v86: f32 = _v39;
    let _v87: f32 = _v45;
    let _v88: f32 = _v51;
    let _v89: vec3<f32> = vec3<f32>(_v86, _v87, _v88);
    let _v90: vec3<f32> = _v89;
    let _v91: f32 = _v57;
    let _v92: f32 = _v62;
    let _v93: f32 = _v68;
    let _v94: vec3<f32> = vec3<f32>(_v91, _v92, _v93);
    let _v95: vec3<f32> = _v94;
    let _v96: f32 = _v74;
    let _v97: f32 = _v80;
    let _v98: f32 = _v85;
    let _v99: vec3<f32> = vec3<f32>(_v96, _v97, _v98);
    let _v100: vec3<f32> = _v99;
    let _v101: mat3x3<f32> = transpose(mat3x3<f32>(_v90, _v95, _v100));
    let _v102: mat3x3<f32> = transpose(_v101);
    let _v103: vec3<f32> = _v102 * p;
    let _v104: vec2<f32> = _v103.xy;
    let _v105: f32 = norm_8(_v104);
    let _v106: f32 = f32(_v10);
    let _v107: f32 = _v105 - _v106;
    let _v108: f32 = _v103.z;
    let _v109: f32 = _v108;
    let _v110: f32 = _v107;
    let _v111: f32 = _v109;
    let _v112: vec2<f32> = vec2<f32>(_v110, _v111);
    let _v113: f32 = norm_8(_v112);
    let _v114: f32 = f32(_v11);
    let _v115: f32 = _v113 - _v114;
    let _v116: f32 = _v12 * _v5;
    let _v117: f32 = max(_v116, _v4);
    let _v118: f32 = _v22 - _v115;
    let _v119: f32 = abs(_v118);
    let _v120: f32 = f32(_v117);
    let _v121: f32 = _v120 - _v119;
    let _v122: f32 = max(_v121, _v3);
    let _v123: f32 = min(_v22, _v115);
    let _v124: f32 = _v122 * _v122;
    let _v125: f32 = _v124 * _v2;
    let _v126: f32 = f32(_v117);
    let _v127: f32 = _v125 / _v126;
    let _v128: f32 = _v123 - _v127;
    let _v129: f32 = norm_0(_v13);
    let _v130: vec3<f32> = vec3<f32>(_v129);
    let _v131: vec3<f32> = _v13 / _v130;
    let _v132: f32 = cos(_v14);
    let _v133: f32 = sin(_v14);
    let _v134: f32 = _v6 - _v132;
    let _v135: f32 = _v131.x;
    let _v136: f32 = _v135;
    let _v137: f32 = _v131.y;
    let _v138: f32 = _v137;
    let _v139: f32 = _v131.z;
    let _v140: f32 = _v139;
    let _v141: f32 = f32(_v134);
    let _v142: f32 = _v141 * _v136;
    let _v143: f32 = _v142 * _v136;
    let _v144: f32 = f32(_v132);
    let _v145: f32 = _v143 + _v144;
    let _v146: f32 = f32(_v134);
    let _v147: f32 = _v146 * _v136;
    let _v148: f32 = _v147 * _v138;
    let _v149: f32 = f32(_v133);
    let _v150: f32 = _v149 * _v140;
    let _v151: f32 = _v148 - _v150;
    let _v152: f32 = f32(_v134);
    let _v153: f32 = _v152 * _v136;
    let _v154: f32 = _v153 * _v140;
    let _v155: f32 = f32(_v133);
    let _v156: f32 = _v155 * _v138;
    let _v157: f32 = _v154 + _v156;
    let _v158: f32 = f32(_v134);
    let _v159: f32 = _v158 * _v136;
    let _v160: f32 = _v159 * _v138;
    let _v161: f32 = f32(_v133);
    let _v162: f32 = _v161 * _v140;
    let _v163: f32 = _v160 + _v162;
    let _v164: f32 = f32(_v134);
    let _v165: f32 = _v164 * _v138;
    let _v166: f32 = _v165 * _v138;
    let _v167: f32 = f32(_v132);
    let _v168: f32 = _v166 + _v167;
    let _v169: f32 = f32(_v134);
    let _v170: f32 = _v169 * _v138;
    let _v171: f32 = _v170 * _v140;
    let _v172: f32 = f32(_v133);
    let _v173: f32 = _v172 * _v136;
    let _v174: f32 = _v171 - _v173;
    let _v175: f32 = f32(_v134);
    let _v176: f32 = _v175 * _v136;
    let _v177: f32 = _v176 * _v140;
    let _v178: f32 = f32(_v133);
    let _v179: f32 = _v178 * _v138;
    let _v180: f32 = _v177 - _v179;
    let _v181: f32 = f32(_v134);
    let _v182: f32 = _v181 * _v138;
    let _v183: f32 = _v182 * _v140;
    let _v184: f32 = f32(_v133);
    let _v185: f32 = _v184 * _v136;
    let _v186: f32 = _v183 + _v185;
    let _v187: f32 = f32(_v134);
    let _v188: f32 = _v187 * _v140;
    let _v189: f32 = _v188 * _v140;
    let _v190: f32 = f32(_v132);
    let _v191: f32 = _v189 + _v190;
    let _v192: f32 = _v145;
    let _v193: f32 = _v151;
    let _v194: f32 = _v157;
    let _v195: vec3<f32> = vec3<f32>(_v192, _v193, _v194);
    let _v196: vec3<f32> = _v195;
    let _v197: f32 = _v163;
    let _v198: f32 = _v168;
    let _v199: f32 = _v174;
    let _v200: vec3<f32> = vec3<f32>(_v197, _v198, _v199);
    let _v201: vec3<f32> = _v200;
    let _v202: f32 = _v180;
    let _v203: f32 = _v186;
    let _v204: f32 = _v191;
    let _v205: vec3<f32> = vec3<f32>(_v202, _v203, _v204);
    let _v206: vec3<f32> = _v205;
    let _v207: mat3x3<f32> = transpose(mat3x3<f32>(_v196, _v201, _v206));
    let _v208: mat3x3<f32> = transpose(_v207);
    let _v209: vec3<f32> = _v208 * p;
    let _v210: vec3<f32> = abs(_v209);
    let _v211: vec3<f32> = _v210 - _v15;
    let _v212: vec3<f32> = vec3<f32>(_v3);
    let _v213: vec3<f32> = max(_v211, _v212);
    let _v214: vec3<f32> = _v213 * _v213;
    let _v215: f32 = _v3 + dot(_v214, vec3<f32>(1.0, 1.0, 1.0));
    let _v216: f32 = _v215 + _v1;
    let _v217: f32 = sqrt(_v216);
    let _v218: f32 = max(_v0, max(max(_v211.x, _v211.y), _v211.z));
    let _v219: f32 = min(_v218, _v3);
    let _v220: f32 = _v217 + _v219;
    let _v221: bool = _v16 > _v3;
    let _v222: f32 = -(_v220);
    let _v223: f32 = -(_v128);
    let _v224: f32 = -(_v222);
    let _v225: f32 = _v16 * _v5;
    let _v226: f32 = max(_v225, _v4);
    let _v227: f32 = _v223 - _v224;
    let _v228: f32 = abs(_v227);
    let _v229: f32 = f32(_v226);
    let _v230: f32 = _v229 - _v228;
    let _v231: f32 = max(_v230, _v3);
    let _v232: f32 = min(_v223, _v224);
    let _v233: f32 = _v231 * _v231;
    let _v234: f32 = _v233 * _v2;
    let _v235: f32 = f32(_v226);
    let _v236: f32 = _v234 / _v235;
    let _v237: f32 = _v232 - _v236;
    let _v238: f32 = -(_v237);
    let _v239: f32 = -(_v220);
    let _v240: f32 = max(_v128, _v239);
    let _v241: f32 = _where(_v221, _v238, _v240);
    let _v242: vec3<f32> = p - _v17;
    let _v243: f32 = norm(_v242);
    let _v244: f32 = f32(_v18);
    let _v245: f32 = _v243 - _v244;
    let _v246: f32 = _v19 * _v5;
    let _v247: f32 = max(_v246, _v4);
    let _v248: f32 = _v241 - _v245;
    let _v249: f32 = abs(_v248);
    let _v250: f32 = f32(_v247);
    let _v251: f32 = _v250 - _v249;
    let _v252: f32 = max(_v251, _v3);
    let _v253: f32 = min(_v241, _v245);
    let _v254: f32 = _v252 * _v252;
    let _v255: f32 = _v254 * _v2;
    let _v256: f32 = f32(_v247);
    let _v257: f32 = _v255 / _v256;
    let _v258: f32 = _v253 - _v257;
    return _v258;
}

struct Uniforms {
  resolution   : vec4<f32>,
  camera_pos   : vec4<f32>,
  camera_target: vec4<f32>,
  light_dir    : vec4<f32>,
  bg_color     : vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

fn sdf_normal(p: vec3<f32>) -> vec3<f32> {
  let e = 0.001;
  return normalize(vec3<f32>(
    sdf(p + vec3<f32>( e, 0.0, 0.0)) - sdf(p + vec3<f32>(-e, 0.0, 0.0)),
    sdf(p + vec3<f32>(0.0,  e, 0.0)) - sdf(p + vec3<f32>(0.0, -e, 0.0)),
    sdf(p + vec3<f32>(0.0, 0.0,  e)) - sdf(p + vec3<f32>(0.0, 0.0, -e)),
  ));
}

fn trace(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
  var t = 0.01;
  for (var i = 0; i < 96; i++) {
    let d = sdf(ro + rd * t);
    if (d < 0.001) { return t; }
    if (t > 100.0) { return -1.0; }
    t += d;
  }
  return -1.0;
}

fn soft_shadow(ro: vec3<f32>, rd: vec3<f32>, k: f32) -> f32 {
  var res = 1.0;
  var t = 0.02;
  for (var i = 0; i < 24; i++) {
    let h = sdf(ro + rd * t);
    if (h < 0.001) { return 0.0; }
    res = min(res, k * h / t);
    t += h;
    if (t > 20.0) { break; }
  }
  return clamp(res, 0.0, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  let pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  return vec4<f32>(pos[vid], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let res = u.resolution.xy;
  let uv = (frag.xy / res - 0.5) * vec2<f32>(res.x / res.y, -1.0);

  let cam = u.camera_pos.xyz;
  let tgt = u.camera_target.xyz;
  let fwd = normalize(tgt - cam);
  let right = normalize(cross(fwd, vec3<f32>(0.0, 1.0, 0.0)));
  let up = cross(right, fwd);

  let ro = cam;
  let rd = normalize(fwd + 1.5 * (uv.x * right + uv.y * up));

  let t = trace(ro, rd);
  var col = u.bg_color.xyz;

  if (t >= 0.0) {
    let pos = ro + rd * t;
    let nor = sdf_normal(pos);
    let ldir = normalize(u.light_dir.xyz);
    let diff = max(dot(nor, ldir), 0.0);
    let sha = soft_shadow(pos + 0.02 * nor, ldir, 8.0);
    let spec = pow(max(dot(reflect(-ldir, nor), -rd), 0.0), 32.0);
    col = vec3<f32>(0.027) + vec3<f32>(0.85) * diff * sha + vec3<f32>(0.4) * spec;
  }

  col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
  return vec4<f32>(col, 1.0);
}
