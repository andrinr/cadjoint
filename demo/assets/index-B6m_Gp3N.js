(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))i(r);new MutationObserver(r=>{for(const a of r)if(a.type==="childList")for(const s of a.addedNodes)s.tagName==="LINK"&&s.rel==="modulepreload"&&i(s)}).observe(document,{childList:!0,subtree:!0});function n(r){const a={};return r.integrity&&(a.integrity=r.integrity),r.referrerPolicy&&(a.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?a.credentials="include":r.crossOrigin==="anonymous"?a.credentials="omit":a.credentials="same-origin",a}function i(r){if(r.ep)return;r.ep=!0;const a=n(r);fetch(r.href,a)}})();const oo=!1,lo=(e,t)=>e===t,zn=Symbol("solid-proxy"),Fa=typeof Proxy=="function",za=Symbol("solid-track"),Nn={equals:lo};let Na=Ha;const St=1,Gn=2,Ga={owned:null,cleanups:null,context:null,owner:null};var pe=null;let oi=null,co=null,he=null,we=null,ht=null,ni=0;function Gt(e,t){const n=he,i=pe,r=e.length===0,a=t===void 0?i:t,s=r?Ga:{owned:null,cleanups:null,context:a?a.context:null,owner:a},o=r?e:()=>e(()=>Be(()=>sn(s)));pe=s,he=null;try{return Ut(o,!0)}finally{he=n,pe=i}}function T(e,t){t=t?Object.assign({},Nn,t):Nn;const n={value:e,observers:null,observerSlots:null,comparator:t.equals||void 0},i=r=>(typeof r=="function"&&(r=r(n.value)),Va(n,r));return[ja.bind(n),i]}function D(e,t,n){const i=mr(e,t,!1,St);un(i)}function Te(e,t,n){Na=po;const i=mr(e,t,!1,St);i.user=!0,ht?ht.push(i):un(i)}function Q(e,t,n){n=n?Object.assign({},Nn,n):Nn;const i=mr(e,t,!0,0);return i.observers=null,i.observerSlots=null,i.comparator=n.equals||void 0,un(i),ja.bind(i)}function uo(e){return Ut(e,!1)}function Be(e){if(he===null)return e();const t=he;he=null;try{return e()}finally{he=t}}function Lr(e,t,n){const i=Array.isArray(e);let r;return a=>{let s;if(i){s=Array(e.length);for(let l=0;l<e.length;l++)s[l]=e[l]()}else s=e();const o=Be(()=>t(s,r,a));return r=s,o}}function pr(e){Te(()=>Be(e))}function ft(e){return pe===null||(pe.cleanups===null?pe.cleanups=[e]:pe.cleanups.push(e)),e}function ja(){if(this.sources&&this.state)if(this.state===St)un(this);else{const e=we;we=null,Ut(()=>Vn(this),!1),we=e}if(he){const e=this.observers;if(!e||e[e.length-1]!==he){const t=e?e.length:0;he.sources?(he.sources.push(this),he.sourceSlots.push(t)):(he.sources=[this],he.sourceSlots=[t]),e?(e.push(he),this.observerSlots.push(he.sources.length-1)):(this.observers=[he],this.observerSlots=[he.sources.length-1])}}return this.value}function Va(e,t,n){let i=e.value;return(!e.comparator||!e.comparator(i,t))&&(e.value=t,e.observers&&e.observers.length&&Ut(()=>{for(let r=0;r<e.observers.length;r+=1){const a=e.observers[r],s=oi&&oi.running;s&&oi.disposed.has(a),(s?!a.tState:!a.state)&&(a.pure?we.push(a):ht.push(a),a.observers&&Ua(a)),s||(a.state=St)}if(we.length>1e6)throw we=[],new Error},!1)),t}function un(e){if(!e.fn)return;sn(e);const t=ni;ho(e,e.value,t)}function ho(e,t,n){let i;const r=pe,a=he;he=pe=e;try{i=e.fn(t)}catch(s){return e.pure&&(e.state=St,e.owned&&e.owned.forEach(sn),e.owned=null),e.updatedAt=n+1,qa(s)}finally{he=a,pe=r}(!e.updatedAt||e.updatedAt<=n)&&(e.updatedAt!=null&&"observers"in e?Va(e,i):e.value=i,e.updatedAt=n)}function mr(e,t,n,i=St,r){const a={fn:e,state:i,updatedAt:null,owned:null,sources:null,sourceSlots:null,cleanups:null,value:t,owner:pe,context:pe?pe.context:null,pure:n};return pe===null||pe!==Ga&&(pe.owned?pe.owned.push(a):pe.owned=[a]),a}function jn(e){if(e.state===0)return;if(e.state===Gn)return Vn(e);if(e.suspense&&Be(e.suspense.inFallback))return e.suspense.effects.push(e);const t=[e];for(;(e=e.owner)&&(!e.updatedAt||e.updatedAt<ni);)e.state&&t.push(e);for(let n=t.length-1;n>=0;n--)if(e=t[n],e.state===St)un(e);else if(e.state===Gn){const i=we;we=null,Ut(()=>Vn(e,t[0]),!1),we=i}}function Ut(e,t){if(we)return e();let n=!1;t||(we=[]),ht?n=!0:ht=[],ni++;try{const i=e();return fo(n),i}catch(i){n||(ht=null),we=null,qa(i)}}function fo(e){if(we&&(Ha(we),we=null),e)return;const t=ht;ht=null,t.length&&Ut(()=>Na(t),!1)}function Ha(e){for(let t=0;t<e.length;t++)jn(e[t])}function po(e){let t,n=0;for(t=0;t<e.length;t++){const i=e[t];i.user?e[n++]=i:jn(i)}for(t=0;t<n;t++)jn(e[t])}function Vn(e,t){e.state=0;for(let n=0;n<e.sources.length;n+=1){const i=e.sources[n];if(i.sources){const r=i.state;r===St?i!==t&&(!i.updatedAt||i.updatedAt<ni)&&jn(i):r===Gn&&Vn(i,t)}}}function Ua(e){for(let t=0;t<e.observers.length;t+=1){const n=e.observers[t];n.state||(n.state=Gn,n.pure?we.push(n):ht.push(n),n.observers&&Ua(n))}}function sn(e){let t;if(e.sources)for(;e.sources.length;){const n=e.sources.pop(),i=e.sourceSlots.pop(),r=n.observers;if(r&&r.length){const a=r.pop(),s=n.observerSlots.pop();i<r.length&&(a.sourceSlots[s]=i,r[i]=a,n.observerSlots[i]=s)}}if(e.tOwned){for(t=e.tOwned.length-1;t>=0;t--)sn(e.tOwned[t]);delete e.tOwned}if(e.owned){for(t=e.owned.length-1;t>=0;t--)sn(e.owned[t]);e.owned=null}if(e.cleanups){for(t=e.cleanups.length-1;t>=0;t--)e.cleanups[t]();e.cleanups=null}e.state=0}function mo(e){return e instanceof Error?e:new Error(typeof e=="string"?e:"Unknown error",{cause:e})}function qa(e,t=pe){throw mo(e)}const Fi=Symbol("fallback");function Hn(e){for(let t=0;t<e.length;t++)e[t]()}function go(e,t,n={}){let i=[],r=[],a=[],s=0,o=t.length>1?[]:null;return ft(()=>Hn(a)),()=>{let l=e()||[],c=l.length,d,h;return l[za],Be(()=>{let p,f,m,w,x,y,_,$,E;if(c===0)s!==0&&(Hn(a),a=[],i=[],r=[],s=0,o&&(o=[])),n.fallback&&(i=[Fi],r[0]=Gt(I=>(a[0]=I,n.fallback())),s=1);else if(s===0){for(r=new Array(c),h=0;h<c;h++)i[h]=l[h],r[h]=Gt(u);s=c}else{for(m=new Array(c),w=new Array(c),o&&(x=new Array(c)),y=0,_=Math.min(s,c);y<_&&i[y]===l[y];y++);for(_=s-1,$=c-1;_>=y&&$>=y&&i[_]===l[$];_--,$--)m[$]=r[_],w[$]=a[_],o&&(x[$]=o[_]);for(p=new Map,f=new Array($+1),h=$;h>=y;h--)E=l[h],d=p.get(E),f[h]=d===void 0?-1:d,p.set(E,h);for(d=y;d<=_;d++)E=i[d],h=p.get(E),h!==void 0&&h!==-1?(m[h]=r[d],w[h]=a[d],o&&(x[h]=o[d]),h=f[h],p.set(E,h)):a[d]();for(h=y;h<c;h++)h in m?(r[h]=m[h],a[h]=w[h],o&&(o[h]=x[h],o[h](h))):r[h]=Gt(u);r=r.slice(0,s=c),i=l.slice(0)}return r});function u(p){if(a[h]=p,o){const[f,m]=T(h);return o[h]=m,t(l[h],f)}return t(l[h])}}}function vo(e,t,n={}){let i=[],r=[],a=[],s=[],o=0,l;return ft(()=>Hn(a)),()=>{const c=e()||[],d=c.length;return c[za],Be(()=>{if(d===0)return o!==0&&(Hn(a),a=[],i=[],r=[],o=0,s=[]),n.fallback&&(i=[Fi],r[0]=Gt(u=>(a[0]=u,n.fallback())),o=1),r;for(i[0]===Fi&&(a[0](),a=[],i=[],r=[],o=0),l=0;l<d;l++)l<i.length&&i[l]!==c[l]?s[l](()=>c[l]):l>=i.length&&(r[l]=Gt(h));for(;l<i.length;l++)a[l]();return o=s.length=a.length=d,i=c.slice(0),r=r.slice(0,o)});function h(u){a[l]=u;const[p,f]=T(c[l]);return s[l]=f,t(p,l)}}}function b(e,t){return Be(()=>e(t||{}))}function fn(){return!0}const zi={get(e,t,n){return t===zn?n:e.get(t)},has(e,t){return t===zn?!0:e.has(t)},set:fn,deleteProperty:fn,getOwnPropertyDescriptor(e,t){return{configurable:!0,enumerable:!0,get(){return e.get(t)},set:fn,deleteProperty:fn}},ownKeys(e){return e.keys()}};function li(e){return(e=typeof e=="function"?e():e)?e:{}}function bo(){for(let e=0,t=this.length;e<t;++e){const n=this[e]();if(n!==void 0)return n}}function Le(...e){let t=!1;for(let s=0;s<e.length;s++){const o=e[s];t=t||!!o&&zn in o,e[s]=typeof o=="function"?(t=!0,Q(o)):o}if(Fa&&t)return new Proxy({get(s){for(let o=e.length-1;o>=0;o--){const l=li(e[o])[s];if(l!==void 0)return l}},has(s){for(let o=e.length-1;o>=0;o--)if(s in li(e[o]))return!0;return!1},keys(){const s=[];for(let o=0;o<e.length;o++)s.push(...Object.keys(li(e[o])));return[...new Set(s)]}},zi);const n={},i=Object.create(null);for(let s=e.length-1;s>=0;s--){const o=e[s];if(!o)continue;const l=Object.getOwnPropertyNames(o);for(let c=l.length-1;c>=0;c--){const d=l[c];if(d==="__proto__"||d==="constructor")continue;const h=Object.getOwnPropertyDescriptor(o,d);if(!i[d])i[d]=h.get?{enumerable:!0,configurable:!0,get:bo.bind(n[d]=[h.get.bind(o)])}:h.value!==void 0?h:void 0;else{const u=n[d];u&&(h.get?u.push(h.get.bind(o)):h.value!==void 0&&u.push(()=>h.value))}}}const r={},a=Object.keys(i);for(let s=a.length-1;s>=0;s--){const o=a[s],l=i[o];l&&l.get?Object.defineProperty(r,o,l):r[o]=l?l.value:void 0}return r}function yo(e,...t){const n=t.length;if(Fa&&zn in e){const r=n>1?t.flat():t[0],a=t.map(s=>new Proxy({get(o){return s.includes(o)?e[o]:void 0},has(o){return s.includes(o)&&o in e},keys(){return s.filter(o=>o in e)}},zi));return a.push(new Proxy({get(s){return r.includes(s)?void 0:e[s]},has(s){return r.includes(s)?!1:s in e},keys(){return Object.keys(e).filter(s=>!r.includes(s))}},zi)),a}const i=[];for(let r=0;r<=n;r++)i[r]={};for(const r of Object.getOwnPropertyNames(e)){let a=n;for(let l=0;l<t.length;l++)if(t[l].includes(r)){a=l;break}const s=Object.getOwnPropertyDescriptor(e,r);!s.get&&!s.set&&s.enumerable&&s.writable&&s.configurable?i[a][r]=s.value:Object.defineProperty(i[a],r,s)}return i}const wo=e=>`Stale read from <${e}>.`;function Ae(e){const t="fallback"in e&&{fallback:()=>e.fallback};return Q(go(()=>e.each,e.children,t||void 0))}function _o(e){const t="fallback"in e&&{fallback:()=>e.fallback};return Q(vo(()=>e.each,e.children,t||void 0))}function G(e){const t=e.keyed,n=Q(()=>e.when,void 0,void 0),i=t?n:Q(n,void 0,{equals:(r,a)=>!r==!a});return Q(()=>{const r=i();if(r){const a=e.children;return typeof a=="function"&&a.length>0?Be(()=>a(t?r:()=>{if(!Be(i))throw wo("Show");return n()})):a}return e.fallback},void 0,void 0)}const xo=["allowfullscreen","async","alpha","autofocus","autoplay","checked","controls","default","disabled","formnovalidate","hidden","indeterminate","inert","ismap","loop","multiple","muted","nomodule","novalidate","open","playsinline","readonly","required","reversed","seamless","selected","adauctionheaders","browsingtopics","credentialless","defaultchecked","defaultmuted","defaultselected","defer","disablepictureinpicture","disableremoteplayback","preservespitch","shadowrootclonable","shadowrootcustomelementregistry","shadowrootdelegatesfocus","shadowrootserializable","sharedstoragewritable"],ko=new Set(["className","value","readOnly","noValidate","formNoValidate","isMap","noModule","playsInline","adAuctionHeaders","allowFullscreen","browsingTopics","defaultChecked","defaultMuted","defaultSelected","disablePictureInPicture","disableRemotePlayback","preservesPitch","shadowRootClonable","shadowRootCustomElementRegistry","shadowRootDelegatesFocus","shadowRootSerializable","sharedStorageWritable",...xo]),So=new Set(["innerHTML","textContent","innerText","children"]),$o=Object.assign(Object.create(null),{className:"class",htmlFor:"for"}),Po=Object.assign(Object.create(null),{class:"className",novalidate:{$:"noValidate",FORM:1},formnovalidate:{$:"formNoValidate",BUTTON:1,INPUT:1},ismap:{$:"isMap",IMG:1},nomodule:{$:"noModule",SCRIPT:1},playsinline:{$:"playsInline",VIDEO:1},readonly:{$:"readOnly",INPUT:1,TEXTAREA:1},adauctionheaders:{$:"adAuctionHeaders",IFRAME:1},allowfullscreen:{$:"allowFullscreen",IFRAME:1},browsingtopics:{$:"browsingTopics",IMG:1},defaultchecked:{$:"defaultChecked",INPUT:1},defaultmuted:{$:"defaultMuted",AUDIO:1,VIDEO:1},defaultselected:{$:"defaultSelected",OPTION:1},disablepictureinpicture:{$:"disablePictureInPicture",VIDEO:1},disableremoteplayback:{$:"disableRemotePlayback",AUDIO:1,VIDEO:1},preservespitch:{$:"preservesPitch",AUDIO:1,VIDEO:1},shadowrootclonable:{$:"shadowRootClonable",TEMPLATE:1},shadowrootdelegatesfocus:{$:"shadowRootDelegatesFocus",TEMPLATE:1},shadowrootserializable:{$:"shadowRootSerializable",TEMPLATE:1},sharedstoragewritable:{$:"sharedStorageWritable",IFRAME:1,IMG:1}});function Eo(e,t){const n=Po[e];return typeof n=="object"?n[t]?n.$:void 0:n}const Io=new Set(["beforeinput","click","dblclick","contextmenu","focusin","focusout","input","keydown","keyup","mousedown","mousemove","mouseout","mouseover","mouseup","pointerdown","pointermove","pointerout","pointerover","pointerup","touchend","touchmove","touchstart"]),Ao=new Set(["altGlyph","altGlyphDef","altGlyphItem","animate","animateColor","animateMotion","animateTransform","circle","clipPath","color-profile","cursor","defs","desc","ellipse","feBlend","feColorMatrix","feComponentTransfer","feComposite","feConvolveMatrix","feDiffuseLighting","feDisplacementMap","feDistantLight","feDropShadow","feFlood","feFuncA","feFuncB","feFuncG","feFuncR","feGaussianBlur","feImage","feMerge","feMergeNode","feMorphology","feOffset","fePointLight","feSpecularLighting","feSpotLight","feTile","feTurbulence","filter","font","font-face","font-face-format","font-face-name","font-face-src","font-face-uri","foreignObject","g","glyph","glyphRef","hkern","image","line","linearGradient","marker","mask","metadata","missing-glyph","mpath","path","pattern","polygon","polyline","radialGradient","rect","set","stop","svg","switch","symbol","text","textPath","tref","tspan","use","view","vkern"]),Mo={xlink:"http://www.w3.org/1999/xlink",xml:"http://www.w3.org/XML/1998/namespace"},X=e=>Q(()=>e());function Co(e,t,n){let i=n.length,r=t.length,a=i,s=0,o=0,l=t[r-1].nextSibling,c=null;for(;s<r||o<a;){if(t[s]===n[o]){s++,o++;continue}for(;t[r-1]===n[a-1];)r--,a--;if(r===s){const d=a<i?o?n[o-1].nextSibling:n[a-o]:l;for(;o<a;)e.insertBefore(n[o++],d)}else if(a===o)for(;s<r;)(!c||!c.has(t[s]))&&t[s].remove(),s++;else if(t[s]===n[a-1]&&n[o]===t[r-1]){const d=t[--r].nextSibling;e.insertBefore(n[o++],t[s++].nextSibling),e.insertBefore(n[--a],d),t[r]=n[a]}else{if(!c){c=new Map;let h=o;for(;h<a;)c.set(n[h],h++)}const d=c.get(t[s]);if(d!=null)if(o<d&&d<a){let h=s,u=1,p;for(;++h<r&&h<a&&!((p=c.get(t[h]))==null||p!==d+u);)u++;if(u>d-o){const f=t[s];for(;o<d;)e.insertBefore(n[o++],f)}else e.replaceChild(n[o++],t[s++])}else s++;else t[s++].remove()}}}const Dr="_$DX_DELEGATE";function To(e,t,n,i={}){let r;return Gt(a=>{r=a,t===document?e():g(t,e(),t.firstChild?null:void 0,n)},i.owner),()=>{r(),t.textContent=""}}function S(e,t,n,i){let r;const a=()=>{const o=i?document.createElementNS("http://www.w3.org/1998/Math/MathML","template"):document.createElement("template");return o.innerHTML=e,n?o.content.firstChild.firstChild:i?o.firstChild:o.content.firstChild},s=t?()=>Be(()=>document.importNode(r||(r=a()),!0)):()=>(r||(r=a())).cloneNode(!0);return s.cloneNode=s,s}function Ce(e,t=window.document){const n=t[Dr]||(t[Dr]=new Set);for(let i=0,r=e.length;i<r;i++){const a=e[i];n.has(a)||(n.add(a),t.addEventListener(a,zo))}}function P(e,t,n){n==null?e.removeAttribute(t):e.setAttribute(t,n)}function Oo(e,t,n,i){i==null?e.removeAttributeNS(t,n):e.setAttributeNS(t,n,i)}function Lo(e,t,n){n?e.setAttribute(t,""):e.removeAttribute(t)}function Lt(e,t){t==null?e.removeAttribute("class"):e.className=t}function Wa(e,t,n,i){if(i)Array.isArray(n)?(e[`$$${t}`]=n[0],e[`$$${t}Data`]=n[1]):e[`$$${t}`]=n;else if(Array.isArray(n)){const r=n[0];e.addEventListener(t,n[0]=a=>r.call(e,n[1],a))}else e.addEventListener(t,n,typeof n!="function"&&n)}function Do(e,t,n={}){const i=Object.keys(t||{}),r=Object.keys(n);let a,s;for(a=0,s=r.length;a<s;a++){const o=r[a];!o||o==="undefined"||t[o]||(Rr(e,o,!1),delete n[o])}for(a=0,s=i.length;a<s;a++){const o=i[a],l=!!t[o];!o||o==="undefined"||n[o]===l||!l||(Rr(e,o,!0),n[o]=l)}return n}function Ro(e,t,n){if(!t)return n?P(e,"style"):t;const i=e.style;if(typeof t=="string")return i.cssText=t;typeof n=="string"&&(i.cssText=n=void 0),n||(n={}),t||(t={});let r,a;for(a in n)t[a]==null&&i.removeProperty(a),delete n[a];for(a in t)r=t[a],r!==n[a]&&(i.setProperty(a,r),n[a]=r);return n}function Oe(e,t,n){n!=null?e.style.setProperty(t,n):e.style.removeProperty(t)}function De(e,t={},n,i){const r={};return i||D(()=>r.children=ln(e,t.children,r.children)),D(()=>typeof t.ref=="function"&&on(t.ref,e)),D(()=>Bo(e,t,n,!0,r,!0)),r}function on(e,t,n){return Be(()=>e(t,n))}function g(e,t,n,i){if(n!==void 0&&!i&&(i=[]),typeof t!="function")return ln(e,t,i,n);D(r=>ln(e,t(),r,n),i)}function Bo(e,t,n,i,r={},a=!1){t||(t={});for(const s in r)if(!(s in t)){if(s==="children")continue;r[s]=Br(e,s,null,r[s],n,a,t)}for(const s in t){if(s==="children")continue;const o=t[s];r[s]=Br(e,s,o,r[s],n,a,t)}}function Fo(e){return e.toLowerCase().replace(/-([a-z])/g,(t,n)=>n.toUpperCase())}function Rr(e,t,n){const i=t.trim().split(/\s+/);for(let r=0,a=i.length;r<a;r++)e.classList.toggle(i[r],n)}function Br(e,t,n,i,r,a,s){let o,l,c,d,h;if(t==="style")return Ro(e,n,i);if(t==="classList")return Do(e,n,i);if(n===i)return i;if(t==="ref")a||n(e);else if(t.slice(0,3)==="on:"){const u=t.slice(3);i&&e.removeEventListener(u,i,typeof i!="function"&&i),n&&e.addEventListener(u,n,typeof n!="function"&&n)}else if(t.slice(0,10)==="oncapture:"){const u=t.slice(10);i&&e.removeEventListener(u,i,!0),n&&e.addEventListener(u,n,!0)}else if(t.slice(0,2)==="on"){const u=t.slice(2).toLowerCase(),p=Io.has(u);if(!p&&i){const f=Array.isArray(i)?i[0]:i;e.removeEventListener(u,f)}(p||n)&&(Wa(e,u,n,p),p&&Ce([u]))}else if(t.slice(0,5)==="attr:")P(e,t.slice(5),n);else if(t.slice(0,5)==="bool:")Lo(e,t.slice(5),n);else if((h=t.slice(0,5)==="prop:")||(c=So.has(t))||!r&&((d=Eo(t,e.tagName))||(l=ko.has(t)))||(o=e.nodeName.includes("-")||"is"in s))h&&(t=t.slice(5),l=!0),t==="class"||t==="className"?Lt(e,n):o&&!l&&!c?e[Fo(t)]=n:e[d||t]=n;else{const u=r&&t.indexOf(":")>-1&&Mo[t.split(":")[0]];u?Oo(e,u,t,n):P(e,$o[t]||t,n)}return n}function zo(e){let t=e.target;const n=`$$${e.type}`,i=e.target,r=e.currentTarget,a=l=>Object.defineProperty(e,"target",{configurable:!0,value:l}),s=()=>{const l=t[n];if(l&&!t.disabled){const c=t[`${n}Data`];if(c!==void 0?l.call(t,c,e):l.call(t,e),e.cancelBubble)return}return t.host&&typeof t.host!="string"&&!t.host._$host&&t.contains(e.target)&&a(t.host),!0},o=()=>{for(;s()&&(t=t._$host||t.parentNode||t.host););};if(Object.defineProperty(e,"currentTarget",{configurable:!0,get(){return t||document}}),e.composedPath){const l=e.composedPath();a(l[0]);for(let c=0;c<l.length-2&&(t=l[c],!!s());c++){if(t._$host){t=t._$host,o();break}if(t.parentNode===r)break}}else o();a(i)}function ln(e,t,n,i,r){for(;typeof n=="function";)n=n();if(t===n)return n;const a=typeof t,s=i!==void 0;if(e=s&&n[0]&&n[0].parentNode||e,a==="string"||a==="number"){if(a==="number"&&(t=t.toString(),t===n))return n;if(s){let o=n[0];o&&o.nodeType===3?o.data!==t&&(o.data=t):o=document.createTextNode(t),n=Rt(e,n,i,o)}else n!==""&&typeof n=="string"?n=e.firstChild.data=t:n=e.textContent=t}else if(t==null||a==="boolean")n=Rt(e,n,i);else{if(a==="function")return D(()=>{let o=t();for(;typeof o=="function";)o=o();n=ln(e,o,n,i)}),()=>n;if(Array.isArray(t)){const o=[],l=n&&Array.isArray(n);if(Ni(o,t,n,r))return D(()=>n=ln(e,o,n,i,!0)),()=>n;if(o.length===0){if(n=Rt(e,n,i),s)return n}else l?n.length===0?Fr(e,o,i):Co(e,n,o):(n&&Rt(e),Fr(e,o));n=o}else if(t.nodeType){if(Array.isArray(n)){if(s)return n=Rt(e,n,i,t);Rt(e,n,null,t)}else n==null||n===""||!e.firstChild?e.appendChild(t):e.replaceChild(t,e.firstChild);n=t}}return n}function Ni(e,t,n,i){let r=!1;for(let a=0,s=t.length;a<s;a++){let o=t[a],l=n&&n[e.length],c;if(!(o==null||o===!0||o===!1))if((c=typeof o)=="object"&&o.nodeType)e.push(o);else if(Array.isArray(o))r=Ni(e,o,l)||r;else if(c==="function")if(i){for(;typeof o=="function";)o=o();r=Ni(e,Array.isArray(o)?o:[o],Array.isArray(l)?l:[l])||r}else e.push(o),r=!0;else{const d=String(o);l&&l.nodeType===3&&l.data===d?e.push(l):e.push(document.createTextNode(d))}}return r}function Fr(e,t,n=null){for(let i=0,r=t.length;i<r;i++)e.insertBefore(t[i],n)}function Rt(e,t,n,i){if(n===void 0)return e.textContent="";const r=i||document.createTextNode("");if(t.length){let a=!1;for(let s=t.length-1;s>=0;s--){const o=t[s];if(r!==o){const l=o.parentNode===e;!a&&!s?l?e.replaceChild(r,o):e.insertBefore(r,n):l&&o.remove()}else a=!0}}else e.insertBefore(r,n);return[r]}const No="http://www.w3.org/2000/svg";function Go(e,t=!1,n=void 0){return t?document.createElementNS(No,e):document.createElement(e,{is:n})}function jo(e,t){const n=Q(e);return Q(()=>{const i=n();switch(typeof i){case"function":return Be(()=>i(t));case"string":const r=Ao.has(i),a=Go(i,r,Be(()=>t.is));return De(a,t,r),a}})}function Vo(e){const[,t]=yo(e,["component"]);return jo(()=>e.component,t)}const Ho="modulepreload",Uo=function(e,t){return new URL(e,t).href},zr={},tt=function(t,n,i){let r=Promise.resolve();if(n&&n.length>0){let s=function(d){return Promise.all(d.map(h=>Promise.resolve(h).then(u=>({status:"fulfilled",value:u}),u=>({status:"rejected",reason:u}))))};const o=document.getElementsByTagName("link"),l=document.querySelector("meta[property=csp-nonce]"),c=l?.nonce||l?.getAttribute("nonce");r=s(n.map(d=>{if(d=Uo(d,i),d in zr)return;zr[d]=!0;const h=d.endsWith(".css"),u=h?'[rel="stylesheet"]':"";if(!!i)for(let m=o.length-1;m>=0;m--){const w=o[m];if(w.href===d&&(!h||w.rel==="stylesheet"))return}else if(document.querySelector(`link[href="${d}"]${u}`))return;const f=document.createElement("link");if(f.rel=h?"stylesheet":Ho,h||(f.as="script"),f.crossOrigin="",f.href=d,c&&f.setAttribute("nonce",c),document.head.appendChild(f),h)return new Promise((m,w)=>{f.addEventListener("load",m),f.addEventListener("error",()=>w(new Error(`Unable to preload CSS for ${d}`)))})}))}function a(s){const o=new Event("vite:preloadError",{cancelable:!0});if(o.payload=s,window.dispatchEvent(o),!o.defaultPrevented)throw s}return r.then(s=>{for(const o of s||[])o.status==="rejected"&&a(o.reason);return t().catch(a)})},Tt=1.5,qo=.05,Wo=200,Nr=100,cn=e=>e.projection==="orthographic",Dt=e=>Tt*e;function gr(e){return e.orthoHeight??Dt(vr(Me(e.target,e.position)))}function Yo(e){return{near:e-Nr,far:e+Nr}}function Ya(e){return cn(e)?Yo(vr(Me(e.target,e.position))):{near:qo,far:Wo}}const Me=(e,t)=>[e[0]-t[0],e[1]-t[1],e[2]-t[2]],ne=(e,t)=>[e[0]+t[0],e[1]+t[1],e[2]+t[2]],K=(e,t)=>[e[0]*t,e[1]*t,e[2]*t],V=(e,t)=>e[0]*t[0]+e[1]*t[1]+e[2]*t[2],at=(e,t)=>[e[1]*t[2]-e[2]*t[1],e[2]*t[0]-e[0]*t[2],e[0]*t[1]-e[1]*t[0]],vr=e=>Math.sqrt(V(e,e));function Ot(e){const t=vr(e);return t<1e-12?[0,0,0]:[e[0]/t,e[1]/t,e[2]/t]}function Gi(e){const t=Math.cos(e.pitch);return[e.target[0]+e.distance*t*Math.sin(e.yaw),e.target[1]-e.distance*t*Math.cos(e.yaw),e.target[2]+e.distance*Math.sin(e.pitch)]}function qt(e,t){const n=Ot(Me(t,e)),i=Math.abs(n[2])>.999?[0,1,0]:[0,0,1],r=Ot(at(n,i)),a=at(r,n);return{forward:n,right:r,up:a}}function Xo(e,t,n){const i=Ya(e),r=i.near,a=i.far,{position:s}=e,{forward:o,right:l,up:c}=qt(s,e.target),d=e.width/e.height;let h;if(cn(e)){const p=gr(e),f=2/(p*d),m=2/p,w=1/(a-r);h=[[f*l[0],f*l[1],f*l[2],-f*V(l,s)],[m*c[0],m*c[1],m*c[2],-m*V(c,s)],[w*o[0],w*o[1],w*o[2],-w*V(o,s)-w*r],[0,0,0,1]]}else{const p=2/(Tt*d),f=2/Tt,m=a/(a-r);h=[[p*l[0],p*l[1],p*l[2],-p*V(l,s)],[f*c[0],f*c[1],f*c[2],-f*V(c,s)],[m*o[0],m*o[1],m*o[2],-m*V(o,s)-m*r],[o[0],o[1],o[2],-V(o,s)]]}const u=new Float32Array(16);for(let p=0;p<4;p++)for(let f=0;f<4;f++)u[p*4+f]=h[f][p];return u}function ye(e,t){const{forward:n,right:i,up:r}=qt(t.position,t.target),a=Me(e,t.position),s=V(a,n);if(!cn(t)&&s<=1e-6)return{x:NaN,y:NaN,viewDepth:s,visible:!1};const o=t.width/t.height,l=cn(t)?gr(t):Tt*s,c=V(a,i)/l,d=V(a,r)/l;return{x:(c/o+.5)*t.width,y:(.5-d)*t.height,viewDepth:s,visible:!0}}function wt(e,t,n){const{forward:i,right:r,up:a}=qt(n.position,n.target),s=n.width/n.height,o=(e/n.width-.5)*s,l=-(t/n.height-.5);if(cn(n)){const d=gr(n),h=ne(K(r,o*d),K(a,l*d));return{origin:ne(n.position,h),direction:i,tMin:Ya(n).near}}const c=Ot(ne(i,K(ne(K(r,o),K(a,l)),Tt)));return{origin:n.position,direction:c,tMin:0}}function Xa(e,t,n){const i=V(e.direction,n);if(Math.abs(i)<1e-6)return null;const r=V(Me(t,e.origin),n)/i;return r<=(e.tMin??0)?null:ne(e.origin,K(e.direction,r))}function Ko(e,t,n,i){const r=Me(e,t);return[V(r,n),V(r,i)]}function hn(e,t,n,i,r,a){const s=r-n,o=a-i,l=s*s+o*o;if(l<1e-12)return Math.hypot(e-n,t-i);const c=Math.max(0,Math.min(1,((e-n)*s+(t-i)*o)/l));return Math.hypot(e-(n+c*s),t-(i+c*o))}function xn(e,t,n,i,r=12,a=!1){let s=null;for(const o of e)if(!(a&&!o.editable))for(let l=0;l<o.vertices.length;l++){const c=ye(o.vertices[l].world,i);if(!c.visible)continue;const d=Math.hypot(c.x-t,c.y-n);d<=r&&(s===null||d<s.distance)&&(s={nodeId:o.id,vertexIndex:l,distance:d})}return s}function ji(e,t,n,i,r=24){let a=null;for(const s of e){if(!s.editable)continue;const o=s.vertices.length,l=s.vertices.map(c=>ye(c.world,i));for(let c=0;c<o;c++){const d=l[c],h=l[(c+1)%o];if(!d.visible||!h.visible)continue;const u=hn(t,n,d.x,d.y,h.x,h.y);u<=r&&(a===null||u<a.distance)&&(a={nodeId:s.id,insertIndex:c+1,distance:u})}}return a}function kn(e,t,n,i,r=10,a=!1){let s=null;for(const o of e)if(o.editable&&!(o.kind==="profile"&&!a))for(const[l,c]of o.edges){const d=ye(l,i),h=ye(c,i);if(!d.visible||!h.visible)continue;const u=hn(t,n,d.x,d.y,h.x,h.y);u<=r&&(s===null||u<s.distance)&&(s={nodeId:o.id,distance:u})}return s}function Gr(e,t,n,i,r=8){let a=null;for(const s of e){if(s.nodeId===null)continue;const o=c=>{c<=r&&(a===null||c<a.distance)&&(a={nodeId:s.nodeId,distance:c})},l=ye(s.origin,i);l.visible&&o(Math.hypot(t-l.x,n-l.y));for(let c=0;c<4;c++){const d=ye(s.corners[c],i),h=ye(s.corners[(c+1)%4],i);!d.visible||!h.visible||o(hn(t,n,d.x,d.y,h.x,h.y))}}return a}function Zo(e,t,n,i){let r=e.vertices.length,a=1/0;for(let s=0;s<e.vertices.length;s++){const o=ye(e.vertices[s].world,i);if(!o.visible)continue;const l=Math.hypot(o.x-t,o.y-n);l<a&&(a=l,r=s+1)}return r}const se={"surface-viewport":"#e6e6e9","surface-base":"#e6e6e9","surface-bar":"#e6e6e9","surface-bar-alt":"#edecee","surface-float":"#f8f7f8","surface-panel":"#f8f7f8","surface-raised":"#f8f7f8","surface-raised-hover":"#edecee",ink:"#1c1814","ink-2":"#4b4642","ink-3":"#635e5a","ink-on-accent":"#0a0a0c",rule:"#b4b2b9","rule-strong":"#848289","rule-heavy":"#4a484f",accent:"#ffb000","accent-press":"#a87400",danger:"#a8341c","danger-ink":"#8a1f10",warn:"#8a5a00",ok:"#00734c",param:"#0065b4","param-ink":"#004a85","axis-x":"#c81e2a","axis-y":"#17742a","axis-z":"#1f5fd2",construction:"#b4138f","construction-strong":"#7a0a61","construction-locked":"#b0559c","viewport-ink":"#1c1814","viewport-ink-2":"#605e65","viewport-mark":"#48464d","viewport-mode":"#605e65","graticule-line":"#adadb3","graticule-axis":"#9c9ca2"};function le(e){const t=e.replace("#",""),n=t.length===3?t.split("").map(i=>i+i).join(""):t;return[0,2,4].map(i=>parseInt(n.slice(i,i+2),16)/255)}const ct={rest:"construction",hover:"construction-strong",selected:"ink",locked:"construction-locked"},mt={rest:"ink-3",hover:"ink-2",selected:"ink",locked:"rule-strong"},Jo=[[.2744554245,.0057679624,.3326638811],[.1077083262,1.3964696839,1.3867705979],[-.3272410968,.2148135645,.0919768808],[-4.5999315182,-5.7582381893,-19.2918089503],[6.2037359013,14.1539649474,56.6562995652],[4.7517868889,-13.7494394044,-65.3209678276],[-5.432077171,4.641571316,26.2721076045]],Qo=[[-.0020666453,-.0006875655,-.0095482507],[.2504864448,.6944550333,2.4952869139],[8.3459009063,-3.5960313696,.3290570684],[-27.6669694889,14.2538530831,-13.6465831585],[52.1706837385,-27.9445843529,12.8810906346],[-50.7585722964,29.0538803789,4.2699357345],[18.6642528253,-11.4900266123,-5.5707689618]],_t=[.902,.902,.914],el=le(se.ink),tl=le(se["surface-float"]),nl=[.3924,.0015,.9925],il=[.5398,.3384,.0033],jr=_t,Vi=["#a0cbe8","#ffbe7d","#ff9d9a","#86bcb6","#8cd17d","#f1ce63","#d4a6c8","#fabfd2","#d7b5a6"].map(le),Un={traced:{token:"ink",label:"Traced"},sampled:{token:"accent-press",label:"Sampled"},undrawn:{token:"danger",label:"Undrawn crossing"},blend:{token:"param",label:"Blend boundary"},tangent:{token:"ok",label:"Tangent"},coincident:{token:"construction-locked",label:"Coincident"},contact:{token:"construction-locked",label:"Contact"},other:{token:"ink-3",label:"Other"}},Ka="danger-ink",Za={edge:{token:ct.rest,alpha:1},edgeHover:{token:ct.hover,alpha:1},edgeSelected:{token:ct.selected,alpha:1},edgeLocked:{token:ct.locked,alpha:1},handle:{token:ct.rest,alpha:1},handleHover:{token:ct.hover,alpha:1},handleSelected:{token:ct.selected,alpha:1},handleLocked:{token:ct.locked,alpha:1},meshWire:{token:"ink-3",alpha:.3},meshSharp:{token:"ink",alpha:.95},faceOutline:{token:"ink",alpha:.9},faceFill:{token:"ink",alpha:.12},faceOutlineLocked:{token:"ink",alpha:.45},faceFillLocked:{token:"ink",alpha:.05},planeFill:{token:"ink-3",alpha:.14},planeOutline:{token:"ink-2",alpha:.9},planeFillHover:{token:"ink-2",alpha:.18},planeOutlineHover:{token:"ink",alpha:.95},planeFillSelected:{token:"ink",alpha:.2},planeOutlineSelected:{token:"ink",alpha:1},planeFillLocked:{token:"rule-strong",alpha:.1},planeOutlineLocked:{token:"rule-strong",alpha:.8},planeFillPreview:{token:"ink-2",alpha:.16},planeOutlinePreview:{token:"ink-2",alpha:.95}};function rl(e){const t=Za[e],[n,i,r]=le(se[t.token]);return[n,i,r,t.alpha]}const Sn=[le(se["axis-x"]),le(se["axis-y"]),le(se["axis-z"])],bt=e=>le(se[e]).map(t=>Math.round(t*255)).join(", "),$t=(e,t)=>`rgba(${bt(e)}, ${t})`;$t("surface-viewport",.94),$t("surface-float",0),$t("ink",0),$t("ink",.1),$t("ink",.22),$t("ink",.42),$t("ink",.09),bt("accent"),bt("ink"),bt("danger"),bt("param"),bt("ok"),bt("viewport-mark"),bt("viewport-mode");const al="Generated from frontend/src/tokens.ts by `npm run tokens`. Do not edit.",je=e=>`vec3<f32>(${e.map(t=>Number(t.toFixed(10))).join(", ")})`;function sl(){return`${[`// ${al.replace("`npm run tokens`","wgslTokens()")}`,`const INK = ${je(le(se.ink))};`,`const INK_2 = ${je(le(se["ink-2"]))};`,`const INK_3 = ${je(le(se["ink-3"]))};`,`const PAPER = ${je(_t)};`,`const SHEET = ${je(le(se["surface-float"]))};`,`const AXIS_X = ${je(Sn[0])};`,`const AXIS_Y = ${je(Sn[1])};`,`const AXIS_Z = ${je(Sn[2])};`,`const EDGE_DARK = ${je(el)};`,`const EDGE_LIGHT = ${je(tl)};`,...Jo.map((t,n)=>`const VIRIDIS_C${n} = ${je(t)};`),...Qo.map((t,n)=>`const MAGMA_C${n} = ${je(t)};`)].join(`
`)}
`}const $n=[[1,0,0],[0,1,0],[0,0,1]],ol=Sn.map(e=>[e[0],e[1],e[2]]),Vr=40;function Hi(e){const[t,n,i]=e,[r,a]=[Math.cos(t),Math.sin(t)],[s,o]=[Math.cos(n),Math.sin(n)],[l,c]=[Math.cos(i),Math.sin(i)];return[[l*s,l*o*a-c*r,l*o*r+c*a],[c*s,c*o*a+l*r,c*o*r-l*a],[-o,s*a,s*r]]}const ll=(e,t)=>[V(e[0],t),V(e[1],t),V(e[2],t)],cl=(e,t)=>[e[0][0]*t[0]+e[1][0]*t[1]+e[2][0]*t[2],e[0][1]*t[0]+e[1][1]*t[1]+e[2][1]*t[2],e[0][2]*t[0]+e[1][2]*t[1]+e[2][2]*t[2]];function Ja(e,t,n,i,r=t.dimensions){const a=Hi(t.rotation),s=Hi(i),o=t.position,l=dl(t.dimensions,r),c=d=>{const h=cl(a,Me(d,o)),u=[h[0]*l[0],h[1]*l[1],h[2]*l[2]];return ne(ll(s,u),n)};return e.map(d=>[c(d[0]),c(d[1])])}function Yt(e,t){return Math.abs(t)<1e-9?1:e/t}function dl(e,t){if(Array.isArray(e.size)&&Array.isArray(t.size))return[Yt(t.size[0],e.size[0]),Yt(t.size[1],e.size[1]),Yt(t.size[2],e.size[2])];if(typeof e.radius=="number"&&typeof t.radius=="number"){const n=Yt(t.radius,e.radius);return typeof e.height=="number"&&typeof t.height=="number"?[n,n,Yt(t.height,e.height)]:[n,n,n]}return[1,1,1]}function ul(e,t,n,i){const r=Object.fromEntries(Object.entries(t).map(([s,o])=>[s,Array.isArray(o)?[...o]:o])),a=Math.max(.05,i);if(e==="box"&&Array.isArray(r.size))r.size[n]*=a;else if(e==="sphere"&&typeof r.radius=="number")r.radius*=a;else if(e==="cylinder"){const s=n===2?"height":"radius";typeof r[s]=="number"&&(r[s]*=a)}return r}const Ui=(e,t)=>Math.max(.15,.18*Math.hypot(...Me(t,e.position)));function br(e){const t=Math.abs(e[1])>.9?[0,0,1]:[0,1,0],n=Ot(at(e,t));return[n,at(e,n)]}function hl(e,t,n){const i=ne(e,K(t,n)),[r,a]=br(t),s=ne(e,K(t,n*.82)),o=n*.07;return[[e,i],[i,ne(s,K(r,o))],[i,ne(s,K(r,-o))],[i,ne(s,K(a,o))],[i,ne(s,K(a,-o))]]}function fl(e,t,n){const[i,r]=br(t),a=n*.78,s=[];for(let o=0;o<Vr;o++){const l=2*Math.PI*o/Vr;s.push(ne(e,ne(K(i,a*Math.cos(l)),K(r,a*Math.sin(l)))))}return s.map((o,l)=>[o,s[(l+1)%s.length]])}function Qa(e,t,n){return $n.map((i,r)=>({axis:r,edges:n==="rotate"?fl(e,i,t):hl(e,i,t)}))}function pl(e,t,n,i,r){const a=ye(n,r),s=ye(i,r);return!a.visible||!s.visible?1/0:hn(e,t,a.x,a.y,s.x,s.y)}function Hr(e,t,n,i,r,a,s=10){let o=null;for(const l of Qa(e,t,n))for(const[c,d]of l.edges){const h=pl(i,r,c,d,a);h<=s&&(o===null||h<o.distance)&&(o={axis:l.axis,distance:h})}return o?.axis??null}function ci(e,t,n){const i=Me(t,e.origin),r=V(n,n),a=V(n,e.direction),s=V(e.direction,e.direction),o=V(n,i),l=V(e.direction,i),c=r*s-a*a;return Math.abs(c)<1e-9?o/(r||1):(a*l-s*o)/c}function Ur(e,t,n){const i=V(e.direction,n);if(Math.abs(i)<1e-4)return null;const r=V(Me(t,e.origin),n)/i;if(r<=0)return null;const a=ne(e.origin,K(e.direction,r)),[s,o]=br(n),l=Me(a,t);return Math.atan2(V(l,o),V(l,s))}function ml(e,t){let n=t-e;for(;n>Math.PI;)n-=2*Math.PI;for(;n<-Math.PI;)n+=2*Math.PI;return n}function gl(e,t){if(!e||e.vertexIndex!==null)return null;const n=t.find(i=>i.id===e.nodeId);if(!n?.transform||!n.editable)return null;if(e.part==="plane"&&n.plane)return{node:n,origin:n.plane.origin};if(n.kind==="profile"&&n.vertices.length>0){const i=n.vertices.reduce((a,s)=>[a[0]+s.world[0],a[1]+s.world[1],a[2]+s.world[2]],[0,0,0]),r=n.vertices.length;return{node:n,origin:[i[0]/r,i[1]/r,i[2]/r]}}return{node:n,origin:n.transform.position}}const di=192,Ze={draft:{label:"Draft",pixelBudget:32e4,maxRatio:1,bounces:3,shadowSamples:1,samples:128,marchSteps:di},high:{label:"High",pixelBudget:9e5,maxRatio:1.25,bounces:6,shadowSamples:2,samples:512,marchSteps:di},ultra:{label:"Ultra",pixelBudget:42e5,maxRatio:2,bounces:8,shadowSamples:4,samples:1024,marchSteps:di}},Pn="ultra",Pt={shadows:1,reflections:2,flat:4,hideSolid:8,hardShadows:16,refineHit:32,section:64};function es(e,t){const n=e.marchSteps;return n===null||!Number.isFinite(n)?t.marchSteps:Math.min(ns,Math.max(ts,Math.round(n)))}const ts=16,ns=512,vl={solid:0,slice:1,gradient:2,normal:3,depth:4},En=e=>e==="slice"||e==="gradient",is=2;function yr(e){return(Math.min(1,Math.max(0,e))*2-1)*is}const jt={projection:"orthographic",shadows:"hard",reflections:!0,flatShading:!0,hideSolid:!1,xray:1,showGraticule:!0,showOverlays:!0,showSketches:!0,showAllPlanes:!1,showMeshEdges:!1,showMeshWireframe:!1,showBrep:!1,showConstraints:!0,showFixedConstraints:!0,showDistanceConstraints:!0,showConstraintValues:!0,sdfView:"solid",sdfAxis:0,sdfFraction:.5,isoOffset:0,marchSteps:null,refineHit:!1,cullBounds:!0,section:!1};function qr(e,t,n){const i=Math.hypot(e,t,n)||1;return{yaw:e===0&&t===0?0:Math.atan2(e,-t),pitch:Math.asin(n/i)}}const bl=[[1,"front","back"],[0,"left","right"],[2,"bottom","top"]];function Et(e){const t=[];for(const[n,i,r]of bl){const a=e[n];a<0?t.push(i):a>0&&t.push(r)}return t.join("-")}const xt=(()=>{const e={},t=qr(1,-1,1);e.iso={...t,label:"ISO"};for(const n of[-1,0,1])for(const i of[-1,0,1])for(const r of[-1,0,1]){if(n===0&&i===0&&r===0)continue;const a=Math.abs(n)+Math.abs(i)+Math.abs(r),s=qr(n,i,r);e[Et([n,i,r])]={...s,...a===3?{label:"ISO"}:{}}}return e})();function rs(e,t,n,i=.01){if(Math.abs(t-n.pitch)>=i)return!1;if(Math.abs(Math.abs(n.pitch)-Math.PI/2)<1e-6)return!0;const r=e-n.yaw;return Math.abs(Math.atan2(Math.sin(r),Math.cos(r)))<i}function yl(e,t){const{shadows:n,reflections:i,flatShading:r,hideSolid:a,refineHit:s,section:o}=e;return(n==="off"?0:Pt.shadows)|(n==="hard"?Pt.hardShadows:0)|(i?Pt.reflections:0)|(r?Pt.flat:0)|(s?Pt.refineHit:0)|(o?Pt.section:0)|(a||t?Pt.hideSolid:0)}function ii(){try{return typeof localStorage>"u"?null:localStorage}catch{return null}}const as="cadjoint.editingMode.v1";function wl(e=ii()){try{const t=e?.getItem(as);return t==="sketch"||t==="simulate"?t:"model"}catch{return"model"}}function _l(e,t=ii()){try{t?.setItem(as,e)}catch{}}const xl="xy";function kl(e){switch(e){case"xy":return[0,0,1];case"yz":return[1,0,0];case"zx":return[0,1,0];default:return null}}function ss(e,t,n){const i=kl(e),r=V(t.direction,i),a=Math.abs(r)>1e-6?ne(t.origin,K(t.direction,-V(t.origin,i)/r)):ne(t.origin,K(t.direction,Math.max(1,n)));return{origin:[a[0],a[1],a[2]],normal:e==="xy"?null:[i[0],i[1],i[2]]}}const Sl=(e,t)=>[V(e[0],t),V(e[1],t),V(e[2],t)],Wr=(e,t)=>[e[0][0]*t[0]+e[1][0]*t[1]+e[2][0]*t[2],e[0][1]*t[0]+e[1][1]*t[1]+e[2][1]*t[2],e[0][2]*t[0]+e[1][2]*t[1]+e[2][2]*t[2]];function $l(e,t,n){const i=V(e,t),r=V(e,e)-n*n,a=i*i-r;if(a<0)return null;const s=Math.sqrt(a),o=-i-s>1e-6?-i-s:-i+s;if(o<=1e-6)return null;const l=ne(e,K(t,o));return{t:o,normal:K(l,1/Math.max(n,1e-9))}}function Pl(e,t,n){let i=-1/0,r=1/0,a=0;for(let c=0;c<3;c++){if(Math.abs(t[c])<1e-9){if(Math.abs(e[c])>n[c])return null;continue}let d=(-n[c]-e[c])/t[c],h=(n[c]-e[c])/t[c];if(d>h&&([d,h]=[h,d]),d>i&&(i=d,a=c),r=Math.min(r,h),i>r)return null}if(r<=1e-6)return null;const s=i>1e-6?i:r,o=ne(e,K(t,s)),l=[0,0,0];return l[a]=Math.sign(o[a])||1,{t:s,normal:l}}function El(e,t,n,i){let r=null;const a=o=>{o&&o.t>1e-6&&(r===null||o.t<r.t)&&(r=o)},s=t[0]*t[0]+t[1]*t[1];if(s>1e-12){const o=e[0]*t[0]+e[1]*t[1],l=e[0]*e[0]+e[1]*e[1]-n*n,c=o*o-s*l;if(c>=0){const d=Math.sqrt(c);for(const h of[(-o-d)/s,(-o+d)/s]){const u=e[2]+t[2]*h;if(Math.abs(u)<=i){const p=ne(e,K(t,h));a({t:h,normal:[p[0]/n,p[1]/n,0]})}}}}if(Math.abs(t[2])>1e-9)for(const o of[1,-1]){const l=(o*i-e[2])/t[2],c=ne(e,K(t,l));c[0]*c[0]+c[1]*c[1]<=n*n&&a({t:l,normal:[0,0,o]})}return r}const ui=e=>typeof e=="number"?e:null;function Il(e,t){let n=null;for(const i of e){const r=i.transform;if(!r||i.kind==="profile")continue;const a=Hi(r.rotation),s=Wr(a,Me(t.origin,r.position)),o=Wr(a,t.direction);let l=null;if(i.kind==="sphere"){const h=ui(r.dimensions.radius);h!==null&&(l=$l(s,o,h))}else if(i.kind==="box"){const h=r.dimensions.size;Array.isArray(h)&&h.length===3&&(l=Pl(s,o,[h[0],h[1],h[2]]))}else if(i.kind==="cylinder"){const h=ui(r.dimensions.radius),u=ui(r.dimensions.height);h!==null&&u!==null&&(l=El(s,o,h,u))}if(!l||n!==null&&l.t>=n.t)continue;const c=ne(t.origin,K(t.direction,l.t));let d=Sl(a,l.normal);V(d,t.direction)>0&&(d=K(d,-1)),n={nodeId:i.id,point:[c[0],c[1],c[2]],normal:[d[0],d[1],d[2]],t:l.t}}return n}const pn=.6,qi=.25,Al=.55,Ml=.12;function Cl(e){const t=Ot(e),n=Math.abs(V(t,[0,1,0]))>.99?[0,0,1]:[0,1,0],i=Ot(at(n,t)),r=at(t,i);return{u:i,v:r}}function os(e,t,n,i,r,a,s){const o=(l,c)=>ne(e,ne(K(t,i+l),K(n,r+c)));return[o(-a,-s),o(a,-s),o(a,s),o(-a,s)]}function Tl(e){const t=e.plane;if(!t)return null;const n=t.origin,i=t.u,r=t.v;let a=1/0,s=-1/0,o=1/0,l=-1/0;for(const f of e.vertices)a=Math.min(a,f.uv[0]),s=Math.max(s,f.uv[0]),o=Math.min(o,f.uv[1]),l=Math.max(l,f.uv[1]);const c=e.vertices.length===0,d=c?0:(a+s)/2,h=c?0:(o+l)/2,u=c?pn:Math.max(pn,(s-a)/2+qi),p=c?pn:Math.max(pn,(l-o)/2+qi);return{nodeId:e.id,label:e.name??e.id,origin:n,u:i,v:r,normal:t.normal,centerU:d,centerV:h,halfU:u,halfV:p,corners:os(n,i,r,d,h,u,p),editable:e.editable&&e.transform!==null,derived:t.reference!==null&&t.reference!==void 0}}function Ol(e,t){const{u:n,v:i}=Cl(t),r=.6+qi;return{nodeId:null,label:"new sketch",origin:e,u:n,v:i,normal:Ot(t),centerU:0,centerV:0,halfU:r,halfV:r,corners:os(e,n,i,0,0,r,r),editable:!0,derived:!1}}function Ll(e){return ne(e.origin,K(e.normal,Al*Math.max(e.halfU,e.halfV)))}function Dl(e){return e.corners[2]}function Rl(e){return Ml*Math.max(e.halfU,e.halfV)}function Bl(e,t){if(!t.showOverlays||!t.showSketches)return[];if(t.showAllPlanes)return[...e];if(!t.sketching)return[];const n=new Set([t.selectedNodeId,t.hoveredNodeId]);return e.filter(i=>n.has(i.nodeId))}function ls(e){const t=[];for(const n of e){const i=Tl(n);i&&t.push(i)}return t}T("");const[be,Fl]=T([]),[zl,Nl]=T([]);T([]);T([]);T([]);T([]);T([]);const[Gl,jl]=T([]),[Vl,Hl]=T(null),[cs,Ul]=T(null),[ds,wr]=T(null);function hi(e){wr(null),Hl(e)}const[Yr,fi]=T(null),[qn]=T(null);T(null);T(null);T({frame:0,playing:!1});T(!1);const[ri,ql]=T(!1),[Em,us]=T(null),[hs,Wn]=T(null),[Wl,pi]=T(null),[fs]=T(null),[In]=T("idle"),[An]=T(null),[Wi,Yl]=T(null),[Yn,Xl]=T(new Set),[ps,Kl]=T(!1),[ms,Zl]=T(!1),[Mn,Yi]=T(null),[Ie,Jl]=T(null);function de(e){wr(null),Jl(e)}const[nt,it]=T(null),[re,Nt]=T("select"),[ai,gs]=T(null),[mi,J]=T({kind:"",text:"Starting…"}),[dn,Ql]=T(""),[Im,ec]=T("");function tc(){ec(dn()),Ql("")}T("");const[nc]=T(!1);T(!1);const[ic,rc]=T(null),[ac,sc]=T(wl()),kt=ac;function oc(e){sc(e),_l(e);{const t=Ie();t&&t.nodeId}}const[Xn]=T(xl),[Xi,gi]=T(null),[_r,Qt]=T(null),[lc,Kn]=T("translate"),[It,yt]=T("object"),[Ki,xr]=T({yaw:xt.iso.yaw,pitch:xt.iso.pitch});function ae(e){return be().find(t=>t.id===e)}function cc(){return ls(Pe())}function mn(){return be().filter(e=>e.kind==="profile")}function Pe(){const e=Xi(),t=_r(),n=be();return t?n.map(i=>i.id===t.nodeId&&i.transform?dc(i,t.position,t.rotation,t.dimensions):i):e?n.map(i=>{if(i.id!==e.nodeId||!i.plane)return i;const{origin:r,u:a,v:s}=i.plane,[o,l]=e.xy,c=[r[0]+a[0]*o+s[0]*l,r[1]+a[1]*o+s[1]*l,r[2]+a[2]*o+s[2]*l],d=i.vertices.map((h,u)=>u===e.vertexIndex?{...h,uv:e.xy,world:c}:h);return{...i,vertices:d}}):n}function dc(e,t,n,i){if(!e.transform)return e;const r=a=>[a[0]+(t[0]-e.transform.position[0]),a[1]+(t[1]-e.transform.position[1]),a[2]+(t[2]-e.transform.position[2])];return{...e,edges:Ja(e.edges,e.transform,t,n,i),vertices:e.vertices.map(a=>({...a,world:r(a.world)})),plane:e.plane?{...e.plane,origin:r(e.plane.origin)}:e.plane,transform:{...e.transform,position:t,rotation:n,dimensions:i}}}function vi(e){const t=Number(e.toFixed(2));return Object.is(t,-0)?"0":String(t)}function uc(e){if(e.kind==="profile"){const r=e.vertices.length;return r>0?`${r} point${r===1?"":"s"}`:null}const t=e.transform?.dimensions;if(!t)return null;const n=[],i=t.size;return Array.isArray(i)&&n.push(i.map(vi).join(" × ")),typeof t.radius=="number"&&n.push(`r ${vi(t.radius)}`),typeof t.height=="number"&&n.push(`h ${vi(t.height)}`),n.length>0?n.join(" · "):null}function Cn(e,t){return`${e} ${t}${e===1?"":"s"}`}function Zi(e,t,n,i,r,a={}){const s=a.reference??!1,o=s?i:t.id;if(e.push({key:i,testKey:o,nodeId:t.id,elementId:null,itemId:a.itemId??null,depth:r,kind:t.kind,label:t.name??t.kind,detail:s?"shared":uc(t),material:t.material,constraintCount:s?0:t.constraints.length,group:!s&&(t.operators.length>0||(a.group??!1)),uses:a.uses??0,reference:s}),!s)for(const l of t.operators){const c=n.find(d=>d.kind==="feature"&&d.call===l.kind&&d.line===l.line);e.push({key:`${i}-op-${l.kind}-${l.line}`,testKey:`${o}-op-${l.kind}-${l.line}`,nodeId:null,elementId:c?.id??null,itemId:null,depth:r+1,kind:"operator",label:l.kind,detail:`line ${l.line}`,material:null,constraintCount:0,group:!1,uses:0,reference:!1})}}const hc={Box:"box",Sphere:"sphere",Cylinder:"cylinder"};function fc(e){const t=new Map;for(const n of e.items)for(const i of n.operands)t.set(i,(t.get(i)??0)+1);for(const n of e.roots)t.set(n,(t.get(n)??0)+1);return t}function vs(e,t,n){if(e.node!==null){const r=n.get(e.node);if(r)return r.name??r.kind}if(e.name!==null)return e.name;const i=e.kind==="transform"?t.get(e.operands[0]):void 0;return i?vs(i,t,n):e.call}function pc(e,t,n,i){const r=e.name??i?.variable??i?.name??null;if(r!==null)return r;const a=e.operands.map(l=>t.get(l)).filter(l=>l!==void 0);if(a.length===0)return e.call;const s=[...new Set(a.map(l=>vs(l,t,n)))],o=s.slice(0,2);return`${e.call} of ${o.join(", ")}${s.length>2?"…":""}`}function mc(e,t){const n=new Set;if(!e)return n;const i=new Map(e.items.map(s=>[s.id,s])),r=new Set,a=[t];for(;a.length>0;){const s=a.pop();if(r.has(s))continue;r.add(s);const o=i.get(s);o&&(o.node!==null&&n.add(o.node),a.push(...o.operands))}return n}function gc(e,t,n,i){const r=new Map(i.items.map(f=>[f.id,f])),a=new Map(e.map(f=>[f.id,f])),s=new Map(t.map(f=>[f.id,f])),o=fc(i),l=new Set(i.items.flatMap(f=>f.node!==null?[f.node]:[])),c=e.filter(f=>!l.has(f.id)),d=l.size+i.items.filter(f=>f.kind==="boolean").length,h=[{key:"scene",testKey:"scene",nodeId:null,elementId:null,itemId:null,depth:0,kind:"scene",label:n,detail:d>0?Cn(d,"object"):"empty",material:null,constraintCount:0,group:i.roots.length>0,uses:0,reference:!1}],u=new Set,p=(f,m,w)=>{const x=r.get(f);if(!x)return;const y=`${m}/${f}`,_=u.has(f);u.add(f);const $=o.get(f)??0,E=x.node!==null?a.get(x.node):void 0;if(E)Zi(h,E,t,y,w,{itemId:f,uses:$,reference:_,group:x.operands.length>0});else{const I=x.element!==null?s.get(x.element):void 0,A=pc(x,r,a,I),B=x.operands.length,N=x.kind==="boolean"?"boolean":x.kind==="transform"?"transform":hc[x.call]??"sdf",L=A.startsWith(x.call)?null:x.call,H=_?"shared":x.kind==="boolean"?[L,Cn(B,"operand")].filter(Boolean).join(" · "):L;h.push({key:y,testKey:_?y:f,nodeId:null,elementId:I?.id??null,itemId:f,depth:w,kind:N,label:A,detail:H,material:null,constraintCount:0,group:!_&&B>0,uses:$,reference:_})}if(!_)for(const I of x.operands)p(I,y,w+1)};for(const f of i.roots)p(f,"scene",1);if(c.length>0){h.push({key:"unused",testKey:"unused",nodeId:null,elementId:null,itemId:null,depth:1,kind:"group",label:"Not in scene",detail:Cn(c.length,"object"),material:null,constraintCount:0,group:!0,uses:0,reference:!1});for(const f of c)Zi(h,f,t,`unused/${f.id}`,2)}return h}function vc(e,t=[],n="scene.py",i=null){if(i&&i.roots.length>0)return gc(e,t,n,i);const r=t.filter(o=>o.kind==="boolean"),a=e.length+r.length,s=[{key:"scene",testKey:"scene",nodeId:null,elementId:null,itemId:null,depth:0,kind:"scene",label:n,detail:a>0?`${a} object${a===1?"":"s"}`:"empty",material:null,constraintCount:0,group:a>0,uses:0,reference:!1}];for(const o of e)Zi(s,o,t,o.id,1);for(const o of r){const l=o.arguments.find(d=>d.name==="operands"),c=l?.text?l.text.split(",").length:0;s.push({key:o.id,testKey:o.id,nodeId:null,elementId:o.id,itemId:null,depth:1,kind:"boolean",label:o.variable??o.name??o.call,detail:`${o.call} · ${Cn(c,"operand")}`,material:null,constraintCount:0,group:!1,uses:0,reference:!1})}return s}function bc(e,t){const n=[];let i=null;for(const r of e)i!==null&&r.depth>i||(i=null,n.push(r),r.group&&t.has(r.key)&&(i=r.depth));return n}var yc=S("<svg><circle r=1.5></svg>",!1,!0,!1),wc=S("<svg><rect width=3 height=3></svg>",!1,!0,!1),kr=S("<svg><path>"),_c=S('<svg><path d="M8.5 1.5L14.5 5V12L8.5 15.5L2.5 12V5ZM2.5 5L8.5 8.5L14.5 5M8.5 8.5V15.5">'),xc=S("<svg><circle cx=8.5 cy=8.5 r=6.5></circle><ellipse cx=8.5 cy=8.5 rx=6.5 ry=2.5>"),kc=S('<svg><path d="M2.5 4.5A6 2 0 0 1 14.5 4.5V12.5A6 2 0 0 1 2.5 12.5Z"></path><path d="M2.5 4.5A6 2 0 0 0 14.5 4.5">'),Sc=S('<svg><path d="M1.5 8.5C3.5 5.2 5.8 3.5 8.5 3.5S13.5 5.2 15.5 8.5C13.5 11.8 11.2 13.5 8.5 13.5S3.5 11.8 1.5 8.5Z"></path><circle cx=8.5 cy=8.5 r=2.25>'),$c=S('<svg><path d="M8.5 2.5L13.5 5.5V11.5L8.5 14.5L3.5 11.5V5.5Z"></path><path d="M3.5 5.5L8.5 8.5L13.5 5.5">'),Pc=S('<svg><path d="M2.5 2.5L13.5 5.5V11.5L2.5 14.5Z"></path><path d="M2.5 8.5H13.5">'),Ec=S('<svg><path d="M2.5 3.5H13.5V13.5H2.5Z"></path><path d="M2.5 8.5H13.5">'),Ic=S('<svg><path d="M8.5 1.5L14.5 5V12L8.5 15.5L2.5 12V5ZM2.5 5L8.5 8.5L14.5 5M8.5 8.5V15.5"></path><path d="M8.5 8.5L2.5 12M8.5 8.5L14.5 12M8.5 8.5V1.5"stroke-dasharray="1.5 1.5"stroke-linecap=butt>'),Ac=S('<svg><path d="M8.5 1.5L14.5 5V12L8.5 15.5L2.5 12V5ZM2.5 5L8.5 8.5L14.5 5M8.5 8.5V15.5"></path><path d="M10.5 8.6V13.6M12.5 7.4V12.4">'),Mc=S('<svg><path d="M8.5 1.5L14.5 5V12L8.5 15.5L2.5 12V5ZM2.5 5L8.5 8.5L14.5 5M8.5 8.5V15.5M8.5 8.5L2.5 12M8.5 8.5L14.5 12M8.5 8.5V1.5">');const _e=16,Ji=1.25,Cc=1.5,Qi=1.5,Fe=e=>({width:e,height:e,viewBox:`0 0 ${_e} ${_e}`,fill:"none",stroke:"currentColor","stroke-width":Ji*_e/e,"stroke-linecap":"square","stroke-linejoin":"miter"}),bi=e=>(()=>{var t=yc();return D(n=>{var i=e.x,r=e.y;return i!==n.e&&P(t,"cx",n.e=i),r!==n.t&&P(t,"cy",n.t=r),n},{e:void 0,t:void 0}),t})(),en=e=>(()=>{var t=wc();return D(n=>{var i=e.x-Qi,r=e.y-Qi;return i!==n.e&&P(t,"x",n.e=i),r!==n.t&&P(t,"y",n.t=r),n},{e:void 0,t:void 0}),t})();function Tc(e,t,n){return e==="point"?Cc+Ji/2:e==="grip"?Qi/Math.max(Math.abs(t),Math.abs(n))+Ji/2:0}const Xr=e=>+e.toFixed(3);function Mt(e,t=null,n=null){const i=e.map(a=>[a[0],a[1]]),r=(a,s,o)=>{const l=i[s][0]-i[a][0],c=i[s][1]-i[a][1],d=Math.hypot(l,c),h=Tc(o,l/d,c/d);i[a]=[i[a][0]+l/d*h,i[a][1]+c/d*h]};return r(0,1,t),r(i.length-1,i.length-2,n),"M"+i.map(a=>`${Xr(a[0])} ${Xr(a[1])}`).join("L")}function Oc(e,t){return e.map((n,i)=>Mt([n,e[(i+1)%e.length]],t,t)).join("")}const Lc=[[8,2.5],[13.5,6.5],[11,13.5],[5,13.5],[2.5,6.5]];function Dc(e){const[t,n,i,r,a]=Lc;return(()=>{var s=kr(),o=s.firstChild;return De(s,Le(()=>Fe(e.size??_e)),!0,!0),g(s,b(bi,{get x(){return t[0]},get y(){return t[1]}}),null),g(s,b(bi,{get x(){return n[0]},get y(){return n[1]}}),null),g(s,b(bi,{get x(){return a[0]},get y(){return a[1]}}),null),D(()=>P(o,"d",Mt([a,t],"point","point")+Mt([t,n],"point","point")+Mt([n,i,r,a],"point","point"))),s})()}function bs(e){return(()=>{var t=_c();return t.firstChild,De(t,Le(()=>Fe(e.size??_e)),!0,!0),t})()}function Rc(e){return(()=>{var t=xc();return De(t,Le(()=>Fe(e.size??_e)),!0,!0),t})()}function Bc(e){return(()=>{var t=kc();return De(t,Le(()=>Fe(e.size??_e)),!0,!0),t})()}function Fc(e){return(()=>{var t=Sc();return De(t,Le(()=>Fe(e.size??_e)),!0,!0),t})()}function er(e){return(()=>{var t=$c();return De(t,Le(()=>Fe(e.size??_e)),!0,!0),t})()}function zc(e){const t=[[3,13],[6,3],[13,9]];return(()=>{var n=kr(),i=n.firstChild;return De(n,Le(()=>Fe(e.size??_e)),!0,!0),g(n,()=>t.map(r=>b(en,{get x(){return r[0]},get y(){return r[1]}})),null),D(()=>P(i,"d",Oc(t,"grip"))),n})()}function Nc(e){const t=[3.5,13.5],n=[3.5,3.5],i=[13.5,3.5],r=[13.5,9.5];return(()=>{var a=kr(),s=a.firstChild;return De(a,Le(()=>Fe(e.size??_e)),!0,!0),g(a,b(en,{get x(){return t[0]},get y(){return t[1]}}),null),g(a,b(en,{get x(){return n[0]},get y(){return n[1]}}),null),g(a,b(en,{get x(){return i[0]},get y(){return i[1]}}),null),g(a,b(en,{get x(){return r[0]},get y(){return r[1]}}),null),D(()=>P(s,"d",Mt([t,n],"grip","grip")+Mt([n,i],"grip","grip")+Mt([i,r],"grip","grip"))),a})()}function Gc(e){return(()=>{var t=Pc();return De(t,Le(()=>Fe(e.size??_e)),!0,!0),t})()}function jc(e){return(()=>{var t=Ec();return De(t,Le(()=>Fe(e.size??_e)),!0,!0),t})()}function Vc(e){return(()=>{var t=Ic(),n=t.firstChild;return n.nextSibling,De(t,Le(()=>Fe(e.size??_e)),!0,!0),t})()}function Hc(e){return(()=>{var t=Ac();return t.firstChild,De(t,Le(()=>Fe(e.size??_e)),!0,!0),t})()}function Uc(e){return(()=>{var t=Mc();return t.firstChild,De(t,Le(()=>Fe(e.size??_e)),!0,!0),t})()}var qc=S("<label><input type=checkbox><span>");const yi=["X","Y","Z"],Wc=(e,t)=>[0,1,2].map(n=>({value:n,label:yi[n],title:t?.(yi[n]),testId:`${e}-${yi[n].toLowerCase()}`}));function ke(e){const t=()=>["switch",e.compact?"compact":null,e.class??null].filter(n=>n!==null).join(" ");return(()=>{var n=qc(),i=n.firstChild,r=i.nextSibling;return i.addEventListener("change",a=>e.onChange(a.currentTarget.checked)),g(r,()=>e.children),D(a=>{var s=t(),o=e.title,l=e.disabled,c=e.testId;return s!==a.e&&Lt(n,a.e=s),o!==a.t&&P(n,"title",a.t=o),l!==a.a&&(i.disabled=a.a=l),c!==a.o&&P(i,"data-testid",a.o=c),a},{e:void 0,t:void 0,a:void 0,o:void 0}),D(()=>i.checked=e.checked),n})()}Ce(["click"]);var Yc=S("<span class=section-count>"),Xc=S("<div class=section-head><b>"),Kc=S("<section class=section>");function Zc(e){return(()=>{var t=Xc(),n=t.firstChild;return g(n,()=>e.title),g(t,b(G,{get when(){return e.count!==void 0},get children(){var i=Yc();return g(i,()=>e.count),i}}),null),g(t,()=>e.children,null),D(()=>P(t,"data-testid",e.testId)),t})()}function Jc(e){return(()=>{var t=Kc();return g(t,b(Zc,{get title(){return e.title},get count(){return e.count},get children(){return e.actions}}),null),g(t,()=>e.children,null),D(()=>P(t,"data-testid",e.testId)),t})()}var Qc=S("<div>"),ed=S("<button type=button>");function Xt(e){return(()=>{var t=Qc();return g(t,b(Ae,{get each(){return e.options},children:n=>(()=>{var i=ed();return i.$$click=()=>e.onSelect(n.value),g(i,()=>n.label),D(r=>{var a=n.value===e.value,s=n.disabled??e.disabled,o=n.title,l=n.testId;return a!==r.e&&i.classList.toggle("active",r.e=a),s!==r.t&&(i.disabled=r.t=s),o!==r.a&&P(i,"title",r.a=o),l!==r.o&&P(i,"data-testid",r.o=l),r},{e:void 0,t:void 0,a:void 0,o:void 0}),i})()})),D(n=>{var i=e.class?`segmented ${e.class}`:"segmented",r=e.testId;return i!==n.e&&Lt(t,n.e=i),r!==n.t&&P(t,"data-testid",n.t=r),n},{e:void 0,t:void 0}),t})()}Ce(["click"]);function ys(e){const t=i=>{const r=e.within();e.open()&&r&&!r.contains(i.target)&&e.close()},n=i=>{i.key!=="Escape"||!e.open()||(i.preventDefault(),i.stopPropagation(),e.close())};window.addEventListener("pointerdown",t,!0),e.escape&&window.addEventListener("keydown",n,!0),ft(()=>{window.removeEventListener("pointerdown",t,!0),window.removeEventListener("keydown",n,!0)})}var td=S("<ul class=ui-select-list role=listbox>"),nd=S("<div><button type=button class=ui-select-field aria-haspopup=listbox><span>"),id=S("<li role=option><i></i><span>");const rd=e=>e.text??(typeof e.label=="string"?e.label:e.value);function Kr(e){const[t,n]=T(!1),[i,r]=T(0);let a,s;const o=Q(()=>e.options.find(f=>f.value===e.value)),l=()=>Math.max(0,e.options.findIndex(f=>f.value===e.value)),c=(f=!0)=>{n(!1),f&&s?.focus()},d=()=>{e.disabled||(r(l()),n(!0))},h=f=>{f.disabled||(c(),f.value!==e.value&&e.onChange(f.value))},u=f=>{const m=e.options.length;if(!m)return;let w=i();for(let x=0;x<m&&(w=(w+f+m)%m,!!e.options[w].disabled);x++);r(w)};ys({open:t,close:()=>c(!1),within:()=>a});const p=f=>{if(!t()){(f.key==="Enter"||f.key===" "||f.key==="ArrowDown")&&(f.preventDefault(),d());return}if(f.key==="Escape")f.preventDefault(),f.stopPropagation(),c();else if(f.key==="ArrowDown"||f.key==="ArrowUp")f.preventDefault(),u(f.key==="ArrowDown"?1:-1);else if(f.key==="Enter"||f.key===" "){f.preventDefault();const m=e.options[i()];m&&h(m)}else if(f.key==="Tab")c(!1);else if(f.key.length===1){const m=f.key.toLowerCase(),w=e.options.length;for(let x=1;x<=w;x++){const y=(i()+x)%w;if(rd(e.options[y]).toLowerCase().startsWith(m)){r(y);break}}}};return(()=>{var f=nd(),m=f.firstChild,w=m.firstChild,x=a;typeof x=="function"?on(x,f):a=f,m.$$keydown=p,m.$$click=()=>t()?c():d();var y=s;return typeof y=="function"?on(y,m):s=m,g(w,()=>o()?.label??o()?.value??e.placeholder??"—"),g(f,b(G,{get when(){return t()},get children(){var _=td();return _.$$keydown=p,g(_,b(Ae,{get each(){return e.options},children:($,E)=>(()=>{var I=id(),A=I.firstChild,B=A.nextSibling;return I.$$click=N=>{N.preventDefault(),h($)},I.addEventListener("pointerenter",()=>!$.disabled&&r(E())),g(A,()=>$.value===e.value?"✓":""),g(B,()=>$.label??$.value),D(N=>{var L=$.value===e.value,H=$.disabled,Z=E()===i(),ie=!!$.disabled,ce=e.testId?`${e.testId}-option-${$.value}`:void 0;return L!==N.e&&P(I,"aria-selected",N.e=L),H!==N.t&&P(I,"aria-disabled",N.t=H),Z!==N.a&&I.classList.toggle("cursor",N.a=Z),ie!==N.o&&I.classList.toggle("disabled",N.o=ie),ce!==N.i&&P(I,"data-testid",N.i=ce),N},{e:void 0,t:void 0,a:void 0,o:void 0,i:void 0}),I})()})),_}}),null),D(_=>{var $=e.class?`ui-select ${e.class}`:"ui-select",E=!!t(),I=e.disabled,A=e.title,B=e["aria-label"],N=t(),L=e.testId,H=e.value;return $!==_.e&&Lt(f,_.e=$),E!==_.t&&f.classList.toggle("open",_.t=E),I!==_.a&&(m.disabled=_.a=I),A!==_.o&&P(m,"title",_.o=A),B!==_.i&&P(m,"aria-label",_.i=B),N!==_.n&&P(m,"aria-expanded",_.n=N),L!==_.s&&P(m,"data-testid",_.s=L),H!==_.h&&P(m,"data-value",_.h=H),_},{e:void 0,t:void 0,a:void 0,o:void 0,i:void 0,n:void 0,s:void 0,h:void 0}),f})()}Ce(["click","keydown"]);var ad=S("<header><span><small>");function Sr(e){return(()=>{var t=ad(),n=t.firstChild,i=n.firstChild;return g(i,()=>e.kicker),g(n,()=>e.title,null),g(t,()=>e.children,null),t})()}Ce(["click"]);new Promise(e=>{});const[sd,od]=T(null);function ld(e){od(e)}function $r(e){return sd()?.kinds[e]??null}function ws(e){const t=$r(e);return t!==null&&!t.available}var cd=S("<span class=tier-tag>"),dd=S("<small class=tier-reason>");function tr(e){const t=()=>$r(e.kind);return b(G,{get when(){return t()},children:n=>(()=>{var i=cd();return g(i,()=>n().label),D(r=>{var a=!n().available,s=n().message??void 0,o=e.testId??`tier-${e.kind}`,l=n().available?"true":"false";return a!==r.e&&i.classList.toggle("tier-public",r.e=a),s!==r.t&&P(i,"title",r.t=s),o!==r.a&&P(i,"data-testid",r.a=o),l!==r.o&&P(i,"data-available",r.o=l),r},{e:void 0,t:void 0,a:void 0,o:void 0}),i})()})}function ud(e){const t=()=>{const n=$r(e.kind);return n&&!n.available?n.message:null};return b(G,{get when(){return t()},get children(){var n=dd();return g(n,t),D(()=>P(n,"data-testid",e.testId??`tier-reason-${e.kind}`)),n}})}var hd=S('<aside class=object-tree-panel data-testid=object-tree-panel><div class=object-tree-rows role=tree aria-label="Scene composition">'),fd=S("<button type=button class=tree-caret tabindex=-1 aria-hidden=true>"),pd=S("<small class=tree-detail>"),md=S('<b class="tree-badge tree-uses">×'),gd=S("<b class=tree-badge>"),vd=S("<small class=tree-material>"),bd=S("<div class=object-tree-row role=treeitem><span class=tree-label>"),yd=S("<i class=tree-caret-space>"),wd=S("<i class=tree-icon aria-hidden=true>");const _d={scene:er,profile:Dc,box:bs,sphere:Rc,cylinder:Bc,operator:null,boolean:er,transform:null,sdf:null,group:null};function xd(){const[e,t]=T(new Set),n=Q(()=>vc(be(),Gl(),ic()??"scene.py",cs())),i=Q(()=>bc(n(),e())),r=l=>{const c=new Set(e());c.has(l)?c.delete(l):c.add(l),t(c)},a=l=>{if(l.nodeId!==null){hi(null),de({nodeId:l.nodeId,vertexIndex:null});return}if((l.kind==="boolean"||l.kind==="transform")&&l.itemId!==null){de(null),hi(l.elementId),wr(l.itemId);return}l.elementId!==null&&(de(null),hi(l.elementId))},s=l=>{if(l.nodeId!==null)return Ie()?.nodeId===l.nodeId;const c=ds();return c!==null?l.itemId===c:l.elementId!==null&&Vl()===l.elementId},o=(l,c,d)=>{const h=u=>{l.currentTarget.closest(".object-tree-rows")?.querySelectorAll(".object-tree-row")?.[u]?.focus()};l.key==="ArrowDown"?(l.preventDefault(),h(Math.min(d+1,i().length-1))):l.key==="ArrowUp"?(l.preventDefault(),h(Math.max(d-1,0))):(l.key==="ArrowLeft"&&c.group&&!e().has(c.key)||l.key==="ArrowRight"&&c.group&&e().has(c.key))&&(l.preventDefault(),r(c.key))};return(()=>{var l=hd(),c=l.firstChild;return g(l,b(Sr,{kicker:"read-only",title:"Object tree"}),c),g(c,b(Ae,{get each(){return i()},children:(d,h)=>(()=>{var u=bd(),p=u.firstChild;return u.addEventListener("mouseleave",()=>d.nodeId!==null&&it(null)),u.addEventListener("mouseenter",()=>d.nodeId!==null&&it({nodeId:d.nodeId,vertexIndex:null})),u.$$keydown=f=>{f.key==="Enter"||f.key===" "?(f.preventDefault(),a(d)):o(f,d,h())},u.$$click=()=>a(d),g(u,b(G,{get when(){return d.group},get fallback(){return yd()},get children(){var f=fd();return f.$$click=m=>{m.stopPropagation(),r(d.key)},g(f,()=>e().has(d.key)?"▸":"▾"),f}}),p),g(u,b(G,{get when(){return _d[d.kind]},children:f=>(()=>{var m=wd();return g(m,b(Vo,{get component(){return f()}})),m})()}),p),g(p,()=>d.label),g(u,b(G,{get when(){return d.detail},get children(){var f=pd();return g(f,()=>d.detail),f}}),null),g(u,b(G,{get when(){return d.uses>1},get children(){var f=md();return f.firstChild,g(f,()=>d.uses,null),D(m=>{var w=`Shared: read by ${d.uses} operands — one object, not ${d.uses}`,x=`tree-uses-${d.testKey}`;return w!==m.e&&P(f,"title",m.e=w),x!==m.t&&P(f,"data-testid",m.t=x),m},{e:void 0,t:void 0}),f}}),null),g(u,b(G,{get when(){return d.constraintCount>0},get children(){var f=gd();return g(f,()=>d.constraintCount),D(()=>P(f,"title",`${d.constraintCount} constraints`)),f}}),null),g(u,b(G,{get when(){return d.material},get children(){var f=vd();return g(f,()=>d.material),f}}),null),D(f=>{var m=!!s(d),w=d.nodeId!==null&&nt()?.nodeId===d.nodeId,x=d.kind==="operator",y=d.kind==="boolean"||d.kind==="transform",_=!!d.reference,$=d.kind==="group",E=d.depth+1,I=s(d),A=d.group?!e().has(d.key):void 0,B=h()===0?0:-1,N=`calc(var(--inset) + var(--unit) * ${4*d.depth})`,L=`tree-row-${d.testKey}`,H=d.itemId??void 0;return m!==f.e&&u.classList.toggle("active",f.e=m),w!==f.t&&u.classList.toggle("hovered",f.t=w),x!==f.a&&u.classList.toggle("operator",f.a=x),y!==f.o&&u.classList.toggle("boolean",f.o=y),_!==f.i&&u.classList.toggle("reference",f.i=_),$!==f.n&&u.classList.toggle("heading",f.n=$),E!==f.s&&P(u,"aria-level",f.s=E),I!==f.h&&P(u,"aria-selected",f.h=I),A!==f.r&&P(u,"aria-expanded",f.r=A),B!==f.d&&P(u,"tabindex",f.d=B),N!==f.l&&Oe(u,"padding-left",f.l=N),L!==f.u&&P(u,"data-testid",f.u=L),H!==f.c&&P(u,"data-item",f.c=H),f},{e:void 0,t:void 0,a:void 0,o:void 0,i:void 0,n:void 0,s:void 0,h:void 0,r:void 0,d:void 0,l:void 0,u:void 0,c:void 0}),u})()})),l})()}Ce(["click","keydown"]);const kd=.22,Bt=1,Ye=1-2*kd,Sd={front:"FRONT",back:"BACK",left:"LEFT",right:"RIGHT",top:"TOP",bottom:"BOTTOM"},nr=e=>{const t=Math.hypot(e[0],e[1],e[2])||1;return K(e,1/t)},wi=e=>K(e.reduce(ne,[0,0,0]),1/e.length);function _i(e,t){const n=e[1].map((r,a)=>r-e[0][a]),i=e[2].map((r,a)=>r-e[1][a]);return V(at(n,i),t)>0?e:[...e].reverse()}function ot(e,t){const n=[0,0,0];return n[e[0]]=t[0],n[e[1]]=t[1],n[e[2]]=t[2],n}const Pr=(()=>{const e=[],t=[0,1,2];for(const n of t)for(const i of[-1,1]){const[r,a]=t.filter(h=>h!==n),s=ot([n,r,a],[i,0,0]),l=_i([[-1,-1],[1,-1],[1,1],[-1,1]].map(([h,u])=>ot([n,r,a],[i*Bt,h*Ye,u*Ye])),s),c=Et(s.map(Math.sign)),d=n===2?[0,1,0]:[0,0,1];e.push({key:c,opposite:Et(K(s,-1).map(Math.sign)),rank:"face",normal:s,polygon:l,centre:wi(l),labelUp:d,labelRight:at(d,s),label:Sd[c]})}for(const n of t)for(const i of t){if(i<=n)continue;const r=t.find(a=>a!==n&&a!==i);for(const a of[-1,1])for(const s of[-1,1]){const o=nr(ot([n,i,r],[a,s,0])),l=_i([ot([n,i,r],[a*Bt,s*Ye,-Ye]),ot([n,i,r],[a*Bt,s*Ye,Ye]),ot([n,i,r],[a*Ye,s*Bt,Ye]),ot([n,i,r],[a*Ye,s*Bt,-Ye])],o);e.push({key:Et(ot([n,i,r],[a,s,0])),opposite:Et(ot([n,i,r],[-a,-s,0])),rank:"edge",normal:o,polygon:l,centre:wi(l),labelRight:[1,0,0],labelUp:[0,0,1],label:""})}}for(const n of[-1,1])for(const i of[-1,1])for(const r of[-1,1]){const a=[n,i,r],s=nr(a),o=_i([0,1,2].map(l=>[0,1,2].map(c=>a[c]*(c===l?Bt:Ye))),s);e.push({key:Et(a),opposite:Et([-n,-i,-r]),rank:"corner",normal:s,polygon:o,centre:wi(o),labelRight:[1,0,0],labelUp:[0,0,1],label:""})}return e})();function $d(e,t){const n=Math.cos(t),i=[n*Math.sin(e),-n*Math.cos(e),Math.sin(t)],r=K(i,-1),a=Math.abs(r[2])>.999?[0,1,0]:[0,0,1],s=nr(at(r,a));return{direction:i,right:s,up:at(s,r)}}function Kt(e,t){return[V(e,t.right),-V(e,t.up)]}const Pd=(e,t)=>V(e.normal,t.direction)>1e-6;function Ed(e){let t=null,n=-1/0;for(const i of Pr){const r=V(i.normal,e.direction);r>n&&(n=r,t=i)}return t}const Zr=[{axis:[1,0,0],label:"X",token:"axis-x"},{axis:[0,1,0],label:"Y",token:"axis-y"},{axis:[0,0,1],label:"Z",token:"axis-z"}];if(Pr.some(e=>xt[e.key]===void 0||xt[e.opposite]===void 0))throw new Error("navCube: a facet names a view the preset table does not have");const Id=["Modes","Tools","Shape","View","Viewport","File","Program","Simulate"],_s=[{command:"cycle-mode",key:"m",label:"M",action:"Cycle Model → Sketch → Simulate",group:"Modes"},{command:"cycle-mode-back",key:"m",shift:!0,label:"⇧M",action:"Cycle modes backwards",group:"Modes"},{command:"mode-model",key:"escape",label:"Esc",action:"Return to Model mode (after everything else is cancelled)",group:"Modes",dispatch:!1},{command:"select-object",key:"o",label:"O",action:"Select whole objects",group:"Tools"},{command:"select-vertex",key:"v",label:"V",action:"Select sketch vertices",group:"Tools"},{command:"tool-sketch",key:"k",label:"K",action:"Start a sketch",group:"Tools"},{command:"tool-polygon",key:"p",label:"P",action:"Add a point to a sketch",group:"Tools"},{command:"tool-box",key:"b",label:"B",action:"Place a box",group:"Tools"},{command:"tool-sphere",key:"s",label:"S",action:"Place a sphere",group:"Tools"},{command:"tool-cylinder",key:"c",label:"C",action:"Place a cylinder",group:"Tools"},{command:"shape-extrude",key:"e",label:"E",action:"Extrude the selected sketch",group:"Shape"},{command:"gizmo-translate",key:"g",label:"G",action:"Move gizmo",group:"Shape"},{command:"gizmo-rotate",key:"r",label:"R",action:"Rotate gizmo",group:"Shape"},{command:"gizmo-scale",key:"t",label:"T",action:"Scale gizmo",group:"Shape"},{command:"view-front",key:"1",label:"1",action:"Front view (Ctrl: back)",group:"View",dispatch:!1},{command:"view-right",key:"3",label:"3",action:"Right view (Ctrl: left)",group:"View",dispatch:!1},{command:"view-top",key:"7",label:"7",action:"Top view (Ctrl: bottom)",group:"View",dispatch:!1},{command:"view-opposite",key:"9",label:"9",action:"Turn to the opposite side",group:"View",dispatch:!1},{command:"view-ortho",key:"5",label:"5",action:"Toggle orthographic / perspective",group:"View",dispatch:!1},{command:"grid-finer",key:"[",label:"[",action:"Zoom in one grid step",group:"View"},{command:"grid-coarser",key:"]",label:"]",action:"Zoom out one grid step",group:"View"},{command:"show-construction",key:"c",shift:!0,label:"⇧C",action:"Toggle the construction overlay",group:"View"},{command:"show-layout-grid",key:"g",shift:!0,label:"⇧G",action:"Toggle the layout grid",group:"View"},{command:"pan-hold",key:" ",label:"Space (hold)",action:"Pan while dragging",group:"Viewport",dispatch:!1},{command:"pan-drag",key:"shift",label:"⇧ drag",action:"Pan the camera",group:"Viewport",dispatch:!1},{command:"grid-wheel",key:"alt",label:"⌥ wheel",action:"Step the grid",group:"Viewport",dispatch:!1},{command:"select-none",key:"escape",label:"Esc",action:"Cancel the tool, then clear the selection",group:"Viewport",dispatch:!1},{command:"edit-delete",key:"backspace",label:"⌫",action:"Delete the selection",group:"Viewport",dispatch:!1},{command:"help-shortcuts",key:"?",shift:"any",label:"?",action:"This sheet",group:"Viewport"},{command:"file-open",key:"o",mod:!0,label:"⌘O",action:"Open a scene (the Scenes window)",group:"File"},{command:"file-save",key:"s",mod:!0,label:"⌘S",action:"Save",group:"File"},{command:"file-save-as",key:"s",mod:!0,shift:!0,label:"⇧⌘S",action:"Save as",group:"File"},{command:"file-export",key:"e",mod:!0,label:"⌘E",action:"Export a mesh or STEP",group:"File"},{command:"program-run",key:"enter",mod:!0,label:"⌘↵",action:"Run the program",group:"Program"},{command:"program-stop",key:".",mod:!0,label:"⌘.",action:"Stop the running compile or solve",group:"Program"},{command:"edit-undo",key:"z",mod:!0,label:"⌘Z",action:"Undo the last source change",group:"Program"},{command:"edit-redo",key:"z",mod:!0,shift:!0,label:"⇧⌘Z",action:"Redo",group:"Program"},{command:"sim-solve",key:"enter",mod:!0,shift:!0,label:"⇧⌘↵",action:"Solve the open (or first) study",group:"Simulate"}];function Ad(e,t){const n=e.metaKey||e.ctrlKey;if(e.altKey||n)return;const i=e.key.toLowerCase(),r=i==="/"&&e.shiftKey?"?":i;return _s.find(a=>a.dispatch!==!1&&!!a.mod==(t==="app")&&a.key===r&&(a.shift==="any"||!!a.shift===e.shiftKey))}const Md={Digit1:{preset:"front",opposite:"back"},Numpad1:{preset:"front",opposite:"back"},Digit3:{preset:"right",opposite:"left"},Numpad3:{preset:"right",opposite:"left"},Digit7:{preset:"top",opposite:"bottom"},Numpad7:{preset:"top",opposite:"bottom"},Digit5:{preset:null,projection:!0},Numpad5:{preset:null,projection:!0},Digit9:{preset:null,reverse:!0},Numpad9:{preset:null,reverse:!0}};Id.map(e=>({title:e,items:_s.filter(t=>t.group===e).map(t=>({keys:t.label,action:t.action}))}));function ir(){const e=document.activeElement;return e&&(e.tagName==="TEXTAREA"||e.tagName==="INPUT"||e.tagName==="SELECT"||e.isContentEditable||e.closest(".cm-editor")!==null)?e:null}const xs=8,Vt=1e3,Tn={minor:.55,major:.8,axis:.7,offPlane:.6},ks=5,Jr={start:1.6,end:7},Cd=["XY","XZ","YZ"],Td=2,Qr=.06;function Od(e,t){const n=Math.cos(t);return[-n*Math.sin(e),n*Math.cos(e),-Math.sin(t)]}function Ld(e){const t=[Math.abs(e[2])*Td,Math.abs(e[1]),Math.abs(e[0])],n=t.map((r,a)=>{const s=Math.max(...t.filter((o,l)=>l!==a));return Dd(-Qr,Qr,r-s)}),i=n.reduce((r,a)=>r+a,0);return i>1e-6?n.map(r=>r/i):[1,0,0]}function Dd(e,t,n){const i=Math.min(1,Math.max(0,(n-e)/(t-e)));return i*i*(3-2*i)}function Rd(e,t){const n=Ld(Od(e,t));let i=0;for(let r=1;r<n.length;r++)n[r]>n[i]&&(i=r);return Cd[i]}const Ss=.4,$s=60,Bd=[1,2,5];function Ps(e,t=2){const n=Math.floor(Math.log10(e)),i=[];for(let r=n-t;r<=n+t;r++)for(const a of Bd)i.push(a*10**r);return i.sort((r,a)=>r-a)}function Fd(e){if(!Number.isFinite(e)||e<=0)return Number.NaN;let t=Number.NaN,n=Number.POSITIVE_INFINITY;for(const i of Ps(e,1)){const r=Math.abs(Math.log(e/i));r<n&&(n=r,t=i)}return t}function zd(e,t){if(!Number.isFinite(e)||e<=0)return Number.NaN;const n=e*1e-9,i=Ps(e);if(t>0)return i.find(a=>a>e+n)??e;const r=i.filter(a=>a<e-n);return r.length?r[r.length-1]:e}function Es(e){return e/xs*Vt}function Nd(e){return e/Vt*xs}function Gd(e){return Nd(e)/Tt}function rr(e,t){return{mm:Fd(Es(Dt(e))),calibrated:t==="orthographic"}}function Is(e){return rr(e,"orthographic").mm/Vt}function Zt(e,t){if(!Number.isFinite(e)||e<=0)return{value:"—",unit:"mm",calibrated:!1,text:"—"};const n=e>=1e3,r=(n?e/1e3:e).toPrecision(3);return{value:r,unit:n?"m":"mm",calibrated:t,text:t?r:`>${r}`}}function As(e){if(!Number.isFinite(e))return{value:"—",unit:"mm"};const t=Math.abs(e)>=1e3,n=t?e/1e3:e,i=Number(n.toPrecision(3));return{value:(t?n.toPrecision(3):String(i)).replace("-","−"),unit:t?"m":"mm"}}function jd(e,t){const n=Math.cos(t);return[["X",n*Math.sin(e)],["Y",-n*Math.cos(e)],["Z",Math.sin(t)]].filter(([,a])=>Math.abs(a)>.001).map(([a,s])=>`${s>0?"+":"−"}${a}`).join("")||"—"}function Vd(e,t){for(const[n,i]of Object.entries(xt))if(rs(e,t,i))return i.label??n.toUpperCase();return"FREE"}const Zn=1.45,ea=.008,ta=.0022;function Hd(e,t,n){return{...e,yaw:e.yaw-t*ea,pitch:Math.max(-Zn,Math.min(Zn,e.pitch+n*ea))}}function Ud(e,t,n){const i=t*ta*e.distance,r=n*ta*e.distance,{target:a}=e,{right:s,up:o}=qt(Gi(e),a);return{...e,target:[a[0]-s[0]*i+o[0]*r,a[1]-s[1]*i+o[1]*r,a[2]-s[2]*i+o[2]*r]}}const qd=e=>Math.max(Ss,Math.min($s,e));function Wd(e,t){return{...e,distance:qd(e.distance*Math.exp(t*.001))}}function Yd(e,t){if(t===0)return e;const n=Es(Dt(e.distance)),i=zd(n,t>0?1:-1),r=Gd(i);return r<Ss||r>$s?e:{...e,distance:r}}var Xd=S('<div class=view-cube data-testid=view-cube><svg class=cube-stage viewBox="-70 -70 140 140"role=group aria-label="View cube"><title>Drag to orbit. Click a face, bevel or corner to snap.'),Kd=S('<svg class=cube-triad viewBox="-32 -32 64 64"role=img aria-label="World axes"data-testid=cube-triad><title>World axes: X red, Y green, Z blue.'),Zd=S("<svg><polygon><title></svg>",!1,!0,!1),Jd=S("<svg><text class=cube-label font-size=0.26 text-anchor=middle dominant-baseline=central></svg>",!1,!0,!1),Qd=S("<svg><line x1=0 y1=0></svg>",!1,!0,!1),eu=S("<svg><text text-anchor=middle dominant-baseline=central></svg>",!1,!0,!1),tu=S('<button type=button class=viewport-bar-button aria-label="Toggle construction skeleton"data-testid=skeleton-toggle><span class=rail-word>Skeleton'),nu=S('<button type=button class=viewport-bar-button aria-label="Toggle projection"data-testid=projection-toggle><span class=rail-word>');const gt=34,na=20,ia=1.32,ra=4,aa=.011,iu=e=>e.map(([t,n])=>`${t.toFixed(2)},${n.toFixed(2)}`).join(" "),xi=e=>e.split("-").map(t=>t[0].toUpperCase()+t.slice(1)).join("-");function ru(e){return`${e.rank==="face"?`${xi(e.key)} view`:e.rank==="edge"?`${xi(e.key)} — 45° edge view`:`Isometric — ${e.key.split("-").join(", ")}`}  ·  Shift-click for ${xi(e.opposite)}`}function au(e){let t=!1,n=0,i=0,r=0;const a=()=>$d(e.yaw,e.pitch),s=()=>{const y=a();return Pr.filter(_=>Pd(_,y)).sort((_,$)=>V(_.normal,y.direction)-V($.normal,y.direction))},o=y=>iu(y.polygon.map(_=>{const[$,E]=Kt(_,a());return[$*gt,E*gt]})),l=y=>rs(e.yaw,e.pitch,xt[y]),c=()=>Ed(a()),d=y=>{const[_,$]=Kt(y,a());return[_*na,$*na]},h=y=>{const _=a(),[$,E]=Kt(y.centre,_),[I,A]=Kt(y.labelRight,_),[B,N]=Kt(y.labelUp,_);return`matrix(${[I*gt,A*gt,-B*gt,-N*gt].map(H=>H.toFixed(3)).join(" ")} ${($*gt).toFixed(2)} ${(E*gt).toFixed(2)})`},u=y=>{t=!0,n=0,i=y.clientX,r=y.clientY},p=y=>{if(!t)return;const _=y.clientX-i,$=y.clientY-r;if(n+=Math.abs(_)+Math.abs($),i=y.clientX,r=y.clientY,n>ra){const E=y.currentTarget;E.hasPointerCapture(y.pointerId)||E.setPointerCapture(y.pointerId)}e.onOrbit(e.yaw-_*aa,Math.max(-Zn,Math.min(Zn,e.pitch+$*aa)))},f=y=>{t=!1;const _=y.currentTarget;_.hasPointerCapture(y.pointerId)&&_.releasePointerCapture(y.pointerId)},m=(y,_)=>{n>ra||e.onPreset(_.shiftKey||_.ctrlKey?y.opposite:y.key)},w=()=>e.onProjection(e.projection==="orthographic"?"perspective":"orthographic"),x=y=>{if(y.metaKey||y.altKey||ir())return;const _=Md[y.code];if(!_)return;if(_.projection){y.preventDefault(),w();return}if(_.reverse){y.preventDefault(),e.onOrbit(e.yaw+Math.PI,-e.pitch);return}const $=y.ctrlKey?_.opposite:_.preset;$&&(y.preventDefault(),e.onPreset($))};return pr(()=>{window.addEventListener("keydown",x),ft(()=>window.removeEventListener("keydown",x))}),[(()=>{var y=Xd(),_=y.firstChild;return _.firstChild,_.addEventListener("pointercancel",f),_.$$pointerup=f,_.$$pointermove=p,_.$$pointerdown=u,g(_,b(Ae,{get each(){return s()},children:$=>(()=>{var E=Zd(),I=E.firstChild;return E.$$click=A=>m($,A),g(I,()=>ru($)),D(A=>{var B=`cube-facet ${$.rank}`,N=!!l($.key),L=c()?.key===$.key,H=o($),Z=`view-${$.key}`,ie=c()?.key===$.key?"true":"false";return B!==A.e&&P(E,"class",A.e=B),N!==A.t&&E.classList.toggle("active",A.t=N),L!==A.a&&E.classList.toggle("front",A.a=L),H!==A.o&&P(E,"points",A.o=H),Z!==A.i&&P(E,"data-testid",A.i=Z),ie!==A.n&&P(E,"data-front",A.n=ie),A},{e:void 0,t:void 0,a:void 0,o:void 0,i:void 0,n:void 0}),E})()}),null),g(_,b(Ae,{get each(){return s().filter($=>$.rank==="face")},children:$=>(()=>{var E=Jd();return g(E,()=>$.label),D(I=>{var A=!!l($.key),B=h($),N=`cube-label-${$.key}`;return A!==I.e&&E.classList.toggle("active",I.e=A),B!==I.t&&P(E,"transform",I.t=B),N!==I.a&&P(E,"data-testid",I.a=N),I},{e:void 0,t:void 0,a:void 0}),E})()}),null),y})(),(()=>{var y=Kd();return y.firstChild,g(y,b(Ae,{each:Zr,children:_=>(()=>{var $=Qd();return D(E=>{var I=`cube-axis ${_.token}`,A=d(_.axis)[0].toFixed(2),B=d(_.axis)[1].toFixed(2);return I!==E.e&&P($,"class",E.e=I),A!==E.t&&P($,"x2",E.t=A),B!==E.a&&P($,"y2",E.a=B),E},{e:void 0,t:void 0,a:void 0}),$})()}),null),g(y,b(Ae,{each:Zr,children:_=>(()=>{var $=eu();return g($,()=>_.label),D(E=>{var I=`cube-axis-label ${_.token}`,A=(d(_.axis)[0]*ia).toFixed(2),B=(d(_.axis)[1]*ia).toFixed(2);return I!==E.e&&P($,"class",E.e=I),A!==E.t&&P($,"x",E.t=A),B!==E.a&&P($,"y",E.a=B),E},{e:void 0,t:void 0,a:void 0}),$})()}),null),y})()]}function su(e){return(()=>{var t=tu(),n=t.firstChild;return t.$$click=()=>e.onToggle(!e.shown),g(t,b(Nc,{size:16}),n),D(i=>{var r=!!e.shown,a=e.shown?"Skeleton shown — click to hide  (⇧C)":"Skeleton hidden — click to show  (⇧C)",s=e.shown;return r!==i.e&&t.classList.toggle("active",i.e=r),a!==i.t&&P(t,"title",i.t=a),s!==i.a&&P(t,"aria-pressed",i.a=s),i},{e:void 0,t:void 0,a:void 0}),t})()}function ou(e){const t=()=>e.projection==="orthographic";return(()=>{var n=nu(),i=n.firstChild;return n.$$click=()=>e.onProjection(t()?"perspective":"orthographic"),g(n,b(G,{get when(){return t()},get fallback(){return b(Gc,{size:16})},get children(){return b(jc,{size:16})}}),i),g(i,()=>t()?"Ortho":"Persp"),D(r=>{var a=!!t(),s=t()?"Orthographic — click for perspective  (5)":"Perspective — click for orthographic  (5)",o=t();return a!==r.e&&n.classList.toggle("active",r.e=a),s!==r.t&&P(n,"title",r.t=s),o!==r.a&&P(n,"aria-pressed",r.a=o),r},{e:void 0,t:void 0,a:void 0}),n})()}Ce(["pointerdown","pointermove","pointerup","click"]);const Ms=e=>{const t=Math.hypot(e[0],e[1],e[2])||1;return[e[0]/t,e[1]/t,e[2]/t]};Ms([-.55,.8,.45]);Ms([.7,.35,-.4]);const ki="application/x-cadjoint-material";Ce(["click","keydown"]);const rt=e=>Number(e.toFixed(3));function lu(e,t,n,i,r=14){let a=null;const s=Math.floor(e.length/3);for(let o=0;o<s;o++){const l=[e[o*3],e[o*3+1],e[o*3+2]],c=ye(l,i);if(!c.visible)continue;const d=Math.hypot(c.x-t,c.y-n);d<=r&&(a===null||d<a.distance)&&(a={index:o,world:[l[0],l[1],l[2]],x:c.x,y:c.y,distance:d})}return a}function cu(e,t){const n=t&&t.spacing.length>0?t.spacing.reduce((i,r)=>i+r,0)/t.spacing.length:.05;return{kind:"sphere",center:[rt(e[0]),rt(e[1]),rt(e[2])],radius:rt(2*n)}}function du(e,t,n){const i=Math.min(t.x0,t.x1),r=Math.max(t.x0,t.x1),a=Math.min(t.y0,t.y1),s=Math.max(t.y0,t.y1),o=[1/0,1/0,1/0],l=[-1/0,-1/0,-1/0];let c=0;const d=Math.floor(e.length/3);for(let u=0;u<d;u++){const p=[e[u*3],e[u*3+1],e[u*3+2]],f=ye(p,n);if(f.visible&&!(f.x<i||f.x>r)&&!(f.y<a||f.y>s)){c++;for(let m=0;m<3;m++)p[m]<o[m]&&(o[m]=p[m]),p[m]>l[m]&&(l[m]=p[m])}}if(c===0)return null;const h=.001;return{kind:"box",min:[rt(o[0]-h),rt(o[1]-h),rt(o[2]-h)],max:[rt(l[0]+h),rt(l[1]+h),rt(l[2]+h)]}}function uu(e,t){return(e?.parameters??[]).some(n=>n.name===t.name&&n.components===t.components)}function Cs(e,t){return!e||e.length===0?"fixed":e.every(n=>uu(t,n))?"free":"unbound"}function hu(e,t){return Cs(e.binding?[e.binding]:null,t)}function Ts(e,t,n){if(Cs(e,n)!=="free")return null;const i={};for(const r of e){const a=r.index===null||r.index===void 0?t.slice(0,r.components):[t[r.index]];if(a.length!==r.components||a.some(s=>!Number.isFinite(s)))return null;i[r.name]=a}return i}function fu(e){return typeof e=="number"?[e]:Array.isArray(e)?[...e]:null}const pu=.001,mu=.5,Os=e=>Math.max(e.tolerance,2*pu);function Er(e,t){const n=Me(t,e.origin);return[V(n,e.xAxis),V(n,e.yAxis)]}function Ls(e,t){const n=t.length;if(n===0)return 1/0;let i=1/0,r=!1;for(let a=0,s=n-1;a<n;s=a++){const o=t[a],l=t[s],c=l[0]-o[0],d=l[1]-o[1],h=e[0]-o[0],u=e[1]-o[1],p=c*c+d*d,f=p<1e-20?0:Math.max(0,Math.min(1,(h*c+u*d)/p)),m=h-c*f,w=u-d*f;if(i=Math.min(i,m*m+w*w),o[1]>e[1]!=l[1]>e[1]){const x=o[0]+(e[1]-o[1])/(l[1]-o[1])*c;e[0]<x&&(r=!r)}}return r?-Math.sqrt(i):Math.sqrt(i)}const Ds=e=>e.polygon.map(t=>Er(e,t));function Rs(e){const t=[];for(const n of e)for(const i of n.faces)t.push({face:i,nodeId:n.id});return t}function sa(e,t,n){let i=null;for(const{face:r,nodeId:a}of Rs(e)){const s=Os(r),o=Math.abs(V(Me(t,r.origin),r.normal));if(o>s||Ls(Er(r,t),Ds(r))>s)continue;const l=V(n,r.normal);l<mu||(i===null||o<i.planeDistance||o===i.planeDistance&&l>i.alignment)&&(i={face:r,nodeId:a,planeDistance:o,alignment:l})}return i}function gu(e,t){let n=null;for(const{face:i,nodeId:r}of Rs(e)){const a=V(t.direction,i.normal);if(Math.abs(a)<1e-9)continue;const s=V(Me(i.origin,t.origin),i.normal)/a;if(s<=(t.tMin??1e-6)||n!==null&&s>=n.t)continue;const o=ne(t.origin,K(t.direction,s)),l=Os(i);if(Ls(Er(i,o),Ds(i))>l)continue;const c=a>0?K(i.normal,-1):i.normal;n={nodeId:r,point:[o[0],o[1],o[2]],normal:[c[0],c[1],c[2]],t:s}}return n}function On(e,t){const n=Il(e,t),i=gu(e,t);return n?i&&i.t<n.t?i:n:i}function oa(e){const t=e.owner?.variable??e.owner?.kind??"solid",[n]=e.reference.args,i=typeof n=="number"?String(n):`'${n}'`;return`${t}.${e.reference.call}(${i})`}const vu=["distance","horizontal","vertical","coincident"],bu=["parallel","perpendicular"];function tn(e){return vu.includes(e)}function Ln(e){return bu.includes(e)}const At={distance:"Distance",horizontal:"Horizontal",vertical:"Vertical",coincident:"Coincident",parallel:"Parallel",perpendicular:"Perpendicular"};function yu(e,t){const n=e-1;return[n,(n+1)%Math.max(t,1)]}function wu(e,t,n,i,r){if(t<=0||n<=0)return{width:t,height:n,fixed:[],distance:[]};const a=t/Math.max(e.width,1),s=n/Math.max(e.height,1),o=u=>{const p=ye(u,e);return{x:p.x*a,y:p.y*s,visible:p.visible}},l=[],c=[],d=(u,p,f,m,w)=>{const x=o(u),y=o(p);if(!x.visible||!y.visible)return;const _=y.x-x.x,$=y.y-x.y,E=Math.hypot(_,$);if(E<2)return;const I=-$/E*17,A=_/E*17,B=x.x+I,N=x.y+A,L=y.x+I,H=y.y+A;c.push({ax:x.x,ay:x.y,bx:y.x,by:y.y,dax:B,day:N,dbx:L,dby:H,labelX:(B+L)*.5,labelY:(N+H)*.5-5,label:Number(f.toPrecision(4)).toString(),key:m,scope:w})};for(const u of i)if(u.kind==="profile")for(let p=0;p<u.constraints.length;p++){const f=u.constraints[p];if(f.kind==="fixed"){const x=u.vertices[f.vertices[0]];if(!x)continue;const y=o(x.world);y.visible&&l.push({x:y.x+9,y:y.y-9,key:`${u.id}-fixed-${p}`,scope:"sketch"});continue}if(f.kind!=="distance")continue;const m=u.vertices[f.vertices[0]],w=u.vertices[f.vertices[1]];!m||!w||typeof f.value!="number"||d(m.world,w.world,f.value,`${u.id}-distance-${p}`,"sketch")}const h=new Map(i.map(u=>[u.id,u]));for(let u=0;u<r.length;u++){const p=r[u];if(p.kind==="fixed"){const w=h.get(p.nodes[0]);if(!w?.transform)continue;const x=o(w.transform.position);x.visible&&l.push({x:x.x+9,y:x.y-9,key:`object-fixed-${u}`,scope:"object"});continue}const f=h.get(p.nodes[0]),m=h.get(p.nodes[1]);!f?.transform||!m?.transform||typeof p.value!="number"||d(f.transform.position,m.transform.position,p.value,`object-distance-${u}`,"object")}return{width:t,height:n,fixed:l,distance:c}}var _u=S('<svg class=constraint-overlay aria-label="Construction constraints"data-testid=constraint-overlay>'),xu=S("<svg><text></svg>",!1,!0,!1),ku=S("<svg><g class=constraint-distance data-testid=constraint-distance-overlay><line></line><line></line><line class=dimension></line><circle r=2.2></circle><circle r=2.2></svg>",!1,!0,!1),Su=S('<svg><g class=constraint-fixed data-testid=constraint-fixed-overlay><circle r=8></circle><rect x=-4 y=-1 width=8 height=6 rx=1.5></rect><path d="M-2.7-1v-2a2.7 2.7 0 0 1 5.4 0v2"></svg>',!1,!0,!1);function $u(e){return b(G,{get when(){return e.show},get children(){var t=_u();return g(t,b(Ae,{get each(){return X(()=>!!e.showDistance)()?e.geometry.distance:[]},children:n=>(()=>{var i=ku(),r=i.firstChild,a=r.nextSibling,s=a.nextSibling,o=s.nextSibling,l=o.nextSibling;return g(i,b(G,{get when(){return e.showValues},get children(){var c=xu();return g(c,()=>n.label),D(d=>{var h=n.labelX,u=n.labelY;return h!==d.e&&P(c,"x",d.e=h),u!==d.t&&P(c,"y",d.t=u),d},{e:void 0,t:void 0}),c}}),null),D(c=>{var d=n.scope,h=n.ax,u=n.ay,p=n.dax,f=n.day,m=n.bx,w=n.by,x=n.dbx,y=n.dby,_=n.dax,$=n.day,E=n.dbx,I=n.dby,A=n.dax,B=n.day,N=n.dbx,L=n.dby;return d!==c.e&&P(i,"data-scope",c.e=d),h!==c.t&&P(r,"x1",c.t=h),u!==c.a&&P(r,"y1",c.a=u),p!==c.o&&P(r,"x2",c.o=p),f!==c.i&&P(r,"y2",c.i=f),m!==c.n&&P(a,"x1",c.n=m),w!==c.s&&P(a,"y1",c.s=w),x!==c.h&&P(a,"x2",c.h=x),y!==c.r&&P(a,"y2",c.r=y),_!==c.d&&P(s,"x1",c.d=_),$!==c.l&&P(s,"y1",c.l=$),E!==c.u&&P(s,"x2",c.u=E),I!==c.c&&P(s,"y2",c.c=I),A!==c.w&&P(o,"cx",c.w=A),B!==c.m&&P(o,"cy",c.m=B),N!==c.f&&P(l,"cx",c.f=N),L!==c.y&&P(l,"cy",c.y=L),c},{e:void 0,t:void 0,a:void 0,o:void 0,i:void 0,n:void 0,s:void 0,h:void 0,r:void 0,d:void 0,l:void 0,u:void 0,c:void 0,w:void 0,m:void 0,f:void 0,y:void 0}),i})()}),null),g(t,b(Ae,{get each(){return X(()=>!!e.showFixed)()?e.geometry.fixed:[]},children:n=>(()=>{var i=Su();return D(r=>{var a=`translate(${n.x} ${n.y})`,s=n.scope;return a!==r.e&&P(i,"transform",r.e=a),s!==r.t&&P(i,"data-scope",r.t=s),r},{e:void 0,t:void 0}),i})()}),null),D(()=>P(t,"viewBox",`0 0 ${e.geometry.width} ${e.geometry.height}`)),t}})}function Pu(e,t,n,i,r){if(t<=0||n<=0)return{width:t,height:n,marks:[]};const a=t/Math.max(e.width,1),s=n/Math.max(e.height,1),o=[],l=(c,d)=>{const h=ye(Dl(c),e);h.visible&&o.push({nodeId:c.nodeId,label:c.label,x:h.x*a,y:h.y*s,derived:c.derived,preview:d})};for(const c of i)l(c,!1);return r&&l(r,!0),{width:t,height:n,marks:o}}var Eu=S('<div class=plane-labels data-testid=plane-labels aria-label="Sketch planes">'),Iu=S("<button type=button class=plane-label>");function Au(e){return b(G,{get when(){return e.show},get children(){var t=Eu();return g(t,b(Ae,{get each(){return e.geometry.marks},children:n=>(()=>{var i=Iu();return i.$$click=()=>{n.nodeId&&de({nodeId:n.nodeId,vertexIndex:null,part:"plane"})},g(i,()=>n.label),D(r=>{var a=!!n.derived,s=!!n.preview,o=`${n.x}px`,l=`${n.y}px`,c=n.preview?"plane-label-preview":"plane-label",d=n.nodeId??void 0,h=n.preview?"Where the new sketch will land":n.derived?`${n.label} — plane derived from a face; edit the reference in the code`:`${n.label} — sketch plane; click to select, drag the gizmo to move`,u=n.preview;return a!==r.e&&i.classList.toggle("derived",r.e=a),s!==r.t&&i.classList.toggle("preview",r.t=s),o!==r.a&&Oe(i,"left",r.a=o),l!==r.o&&Oe(i,"top",r.o=l),c!==r.i&&P(i,"data-testid",r.i=c),d!==r.n&&P(i,"data-node",r.n=d),h!==r.s&&P(i,"title",r.s=h),u!==r.h&&(i.disabled=r.h=u),r},{e:void 0,t:void 0,a:void 0,o:void 0,i:void 0,n:void 0,s:void 0,h:void 0}),i})()})),t}})}Ce(["click"]);var Mu=S("<dt>"),Cu=S("<small> "),Tu=S("<dd>"),Ou=S('<dl class="title-block perf-readout"data-testid=perf-readout>');const Lu=500;function Ft(e){return[(()=>{var t=Mu();return g(t,()=>e.label),t})(),(()=>{var t=Tu();return g(t,()=>e.value,null),g(t,b(G,{get when(){return e.unit},get children(){var n=Cu();return n.firstChild,g(n,()=>e.unit,null),n}}),null),t})()]}function Du(e){const[t,n]=T(e.renderer.frameStats),i=window.setInterval(()=>n(e.renderer.frameStats),Lu);ft(()=>clearInterval(i));const r=()=>{const a=qn();return a?{elements:a.info?.elements??null,triangles:Math.floor(a.payload.indices.length/3)}:null};return b(G,{get when(){return e.show},get children(){var a=Ou();return g(a,b(Ft,{label:"FPS",get value(){return X(()=>t().fps===null)()?"idle":String(t().fps)}}),null),g(a,b(Ft,{label:"CPU",get value(){return X(()=>t().cpuMs===null)()?"—":t().cpuMs.toFixed(1)},get unit(){return t().cpuMs===null?void 0:"ms/frame"}}),null),g(a,b(Ft,{label:"RES",get value(){return`${t().width} × ${t().height}`},unit:"px"}),null),g(a,b(Ft,{label:"NODES",get value(){return String(be().length)}}),null),g(a,b(G,{get when(){return r()},children:s=>[b(Ft,{label:"ELEMENTS",get value(){return X(()=>s().elements===null)()?"—":s().elements.toLocaleString("en-US")}}),b(Ft,{label:"TRIS",get value(){return s().triangles.toLocaleString("en-US")}})]}),null),a}})}const Bs=e=>`rgb(${e.map(t=>Math.round(Math.min(1,Math.max(0,t))*255)).join(", ")})`;function Ru(e,t=12){const n=[];for(let i=0;i<=t;i++)n.push(Bs(e(i/t)));return`linear-gradient(to right, ${n.join(", ")})`}function Bu(e){const t=Math.min(1,Math.max(-1,e)),n=t<0?nl:il,i=Math.sqrt(Math.abs(t));return[0,1,2].map(r=>jr[r]+(n[r]-jr[r])*i)}function Fs(e=16){return Ru(t=>Bu(t*2-1),e)}var Fu=S('<div class=graticule-gain data-testid=graticule-gain title="Construction grid spacing, on a 1-2-5 ladder, and the world plane it is ruled on: the floor XY wherever the floor is worth drawing, the wall the camera faces in the shallow views where it is not. Scroll to zoom freely; hold Alt to zoom in detents. A > prefix means an on-screen measurement is not to scale."><span>GRID</span><b></b><i> · </i><span>VIEW</span><b></b><i>· '),zu=S(`<div class="graticule-gain graticule-contour plated"data-testid=sdf-contour-readout title="The cutting plane, and the contour intervals on it: major lines at the floor grid's spacing, minor lines at a fifth of it, tapering out two major intervals either side of the surface. A tier stops being drawn once one interval is thinner than a few pixels."><span>SLICE</span><b></b><i>· <!> </i><span>MAJOR</span><b></b><i></i><span>MINOR</span><b></b><i>`),Nu=S("<div class=sdf-key data-testid=sdf-key><div class=sdf-key-ramp></div><div class=sdf-key-values><span></span><span></span><span>"),Gu=S('<div class="graticule-gain graticule-contour plated"data-testid=sdf-depth-readout title="Linear depth along the primary ray, across the framed volume — half a frame either side of the orbit target, so the part fills the ramp at any zoom."><span>DEPTH</span><b></b><i> near</i><b></b><i> far'),ju=S('<div class="graticule-gain graticule-contour plated"data-testid=sdf-normal-readout title="World-space surface normals, encoded n × 0.5 + 0.5."><span>NORMAL</span><b>n × 0.5 + 0.5</b><i>· +X red · +Y green · +Z blue');function Vu(e){const t=Q(()=>{const{mm:a,calibrated:s}=rr(e.camera.distance,e.camera.projection);return Zt(a,s)}),n=Q(()=>{const{mm:a}=rr(e.camera.distance,e.camera.projection);return{major:Zt(a,!0),minor:Zt(a/ks,!0),full:Zt(a*2,!0)}}),i=Q(()=>{const a=Dt(e.camera.distance),s=Math.max(e.camera.distance-a/2,.001),o=e.camera.distance+a/2,l=c=>Zt(c*Vt,!0);return{near:l(s),far:l(o)}}),r=Q(()=>({axis:["X","Y","Z"][e.sdfAxis],at:As(yr(e.sdfFraction)*Vt)}));return[b(G,{get when(){return e.show},get children(){var a=Fu(),s=a.firstChild,o=s.nextSibling,l=o.nextSibling,c=l.firstChild,d=l.nextSibling,h=d.nextSibling,u=h.nextSibling;return u.firstChild,g(o,()=>t().text),g(l,()=>t().unit,c),g(l,()=>Rd(e.camera.yaw,e.camera.pitch),null),g(h,()=>Vd(e.camera.yaw,e.camera.pitch)),g(u,()=>jd(e.camera.yaw,e.camera.pitch),null),D(()=>a.classList.toggle("plated",e.sdfView!=="solid")),a}}),b(G,{get when(){return En(e.sdfView)},get children(){return[(()=>{var a=zu(),s=a.firstChild,o=s.nextSibling,l=o.nextSibling,c=l.firstChild,d=c.nextSibling;d.nextSibling;var h=l.nextSibling,u=h.nextSibling,p=u.nextSibling,f=p.nextSibling,m=f.nextSibling,w=m.nextSibling;return g(o,()=>r().axis),g(l,()=>r().at.value,d),g(l,()=>r().at.unit,null),g(u,()=>n().major.text),g(p,()=>n().major.unit),g(m,()=>n().minor.text),g(w,()=>n().minor.unit),a})(),(()=>{var a=Nu(),s=a.firstChild,o=s.nextSibling,l=o.firstChild,c=l.nextSibling,d=c.nextSibling;return g(l,(()=>{var h=X(()=>e.sdfView==="gradient");return()=>h()?"0.5":`−${n().full.text} ${n().full.unit}`})()),g(c,()=>e.sdfView==="gradient"?"|∇f| 1.0":"f = 0"),g(d,(()=>{var h=X(()=>e.sdfView==="gradient");return()=>h()?"1.5":`+${n().full.text} ${n().full.unit}`})()),D(h=>Oe(s,"background",Fs())),a})()]}}),b(G,{get when(){return e.sdfView==="depth"},get children(){var a=Gu(),s=a.firstChild,o=s.nextSibling,l=o.nextSibling,c=l.firstChild,d=l.nextSibling,h=d.nextSibling,u=h.firstChild;return g(o,()=>i().near.text),g(l,()=>i().near.unit,c),g(d,()=>i().far.text),g(h,()=>i().far.unit,u),a}}),b(G,{get when(){return e.sdfView==="normal"},get children(){return ju()}})]}function ar(e,t,n){const i=t?.transform;if(!i)return null;if(e.mode==="translate")return{argument:i.positionArgument,value:[...n.position]};if(e.mode==="rotate")return{argument:"rotation",value:[...n.rotation]};const r=t.kind==="box"?"size":t.kind==="cylinder"&&e.axis===2?"height":"radius",a=fu(n.dimensions[r]);return a===null?null:{argument:r,value:a}}function Hu(e){const t=(s,o)=>{const l=ae(s.nodeId),c=ar(s,l,o),d=c?Ts(l?.transform?.bindings?.[c.argument],c.value,e.renderer.parameterProgram):null;return d?e.renderer.setParameterOverrides(d):!1};return{hoverAxis:(s,o,l)=>{const c=e.renderer.gizmoTarget();if(!c)return!1;const d=e.renderer.gizmoModeFor(c.node),h=Hr(c.origin,Ui(l,c.origin),d,s,o,l);return h!==e.renderer.gizmoAxis&&(e.renderer.gizmoAxis=h,e.renderer.setConstruction(Pe(),Ie(),nt())),h!==null},begin:(s,o)=>{const l=e.renderer.gizmoTarget();if(!l)return null;const c=e.pickView(),d=Ui(c,l.origin),h=e.renderer.gizmoModeFor(l.node),u=Hr(l.origin,d,h,s,o,c);if(u===null)return null;const p=l.node.transform,f=wt(s,o,c),m=h==="rotate"?Ur(f,l.origin,$n[u]):ci(f,l.origin,$n[u]);return m===null?null:{kind:"gizmo",nodeId:l.node.id,axis:u,mode:h,start:m,position:[...p.position],rotation:[...p.rotation],dimensions:Object.fromEntries(Object.entries(p.dimensions).map(([w,x])=>[w,Array.isArray(x)?[...x]:x])),scaleLength:d,gizmoOrigin:[...l.origin],moved:!1}},update:(s,o,l)=>{const c=e.pickView(),d=wt(o,l,c),h=s.gizmoOrigin,u=$n[s.axis];if(s.mode==="translate"){const f=ci(d,h,u)-s.start,m=ne(s.position,K(u,f));s.moved=!0;const w={position:m,rotation:s.rotation,dimensions:s.dimensions};Qt({...s,...w}),t(s,w)}else if(s.mode==="rotate"){const p=Ur(d,h,u);if(p===null)return;const f=[...s.rotation];f[s.axis]+=ml(s.start,p),s.moved=!0;const m={position:s.position,rotation:f,dimensions:s.dimensions};Qt({...s,...m}),t(s,m)}else{const f=1+(ci(d,h,u)-s.start)/s.scaleLength,m=ae(s.nodeId);if(!m)return;const w=ul(m.kind,s.dimensions,s.axis,f);s.moved=!0;const x={position:s.position,rotation:s.rotation,dimensions:w};Qt({...s,...x}),t(s,x)}},commit:async s=>{const o=_r(),l=ae(s.nodeId);Qt(null),e.renderer.gizmoAxis=null;const c=l?.transform,d=o?ar(s,l,o):null;s.moved&&d&&c&&c.line!==null?(await e.props.onSetValue(c.line,c.call,d.argument,d.argument==="radius"||d.argument==="height"?d.value[0]:d.value),e.renderer.dropStaleParameterOverrides()):s.moved&&e.renderer.setParameterOverrides(null)}}}function Uu(e){const t=()=>kt()==="simulate"&&qn()!==null&&e.renderer.simulationActive,n=(a,s)=>{const o=e.canvas();return{x:a*o.clientWidth/Math.max(o.width,1),y:s*o.clientHeight/Math.max(o.height,1)}};return{simInteractive:t,toCss:n,handleSimTap:(a,s)=>{const o=qn();if(!o)return;const l=lu(o.payload.positions,a,s,e.pickView());if(!l){Wn(null);return}if(ri()){us(cu(l.world,o.info?.grid??null));return}const c=n(l.x,l.y);Wn({x:c.x,y:c.y,world:l.world,value:o.scalars[l.index]??0,label:o.fieldLabel})},rectFromGesture:a=>{const s=n(a.x0,a.y0),o=n(a.x1,a.y1);return{left:Math.min(s.x,o.x),top:Math.min(s.y,o.y),width:Math.abs(o.x-s.x),height:Math.abs(o.y-s.y)}}}}const Dn=new Map,[Am,la]=T(0);function qu(e){for(const[t,n]of Object.entries(e))Dn.set(t,n);return la(t=>t+1),()=>{for(const[t,n]of Object.entries(e))Dn.get(t)===n&&Dn.delete(t);la(t=>t+1)}}function Wu(e){const t=Dn.get(e);return!t||(t.blocked?.()??null)!==null?!1:(t.run(),!0)}function Yu(e){return e.gesture?"gesture":e.pendingConstraint||e.pendingLoft?"pending":e.toolArmed||e.bcPickArmed?"tool":e.selection||e.simProbe?"selection":e.awayFromModel?"mode":null}function Xu(e){return{onKeyDown:i=>{const r=ir()!==null;if(i.key==="Escape"&&!document.querySelector(".dialog-backdrop, .menu-dropdown, .tool-group.open")){const o=Yu({gesture:e.gestureActive(),pendingConstraint:e.pendingConstraintActive(),pendingLoft:ai()!==null,toolArmed:re()!=="select",bcPickArmed:ri(),selection:Ie()!==null,simProbe:hs()!==null,awayFromModel:kt()!=="model"});o!==null&&i.preventDefault(),o==="gesture"?e.cancelGesture():o==="pending"?(e.clearPendingConstraint(),gs(null)):o==="tool"?(Nt("select"),ql(!1)):o==="selection"?(de(null),Wn(null)):o==="mode"&&oc("model")}if(!r){const s=Ad(i,"viewport");if(s){i.preventDefault(),Wu(s.command);return}}const a=Ie();if((i.key==="Delete"||i.key==="Backspace")&&a){if(r)return;const s=ae(a.nodeId);s?.editable&&s.line!==null&&(i.preventDefault(),a.vertexIndex!==null?e.props.onPatch("delete_vertex",s.line,a.vertexIndex):e.props.onDeleteObject(s.line),de(null))}},onPanKey:i=>{i.code==="Space"&&(ir()||(i.preventDefault(),e.setPanHeld(i.type==="keydown")))}}}function Ku(e,t){return!t||t.kind!=="profile"?"Loft: click a second sketch.":t.id===e.nodeId?"Loft: choose a different sketch.":!t.editable||t.line===null?"That sketch cannot be edited from source.":t.operators.length>0?"That sketch already drives an operation.":null}function Zu(e){return{handlePlacePrimitive:async(c,d,h)=>{const u=e.pickView(),p=wt(d,h,u),f=Xa(p,[0,0,0],[0,0,1])??ne(p.origin,K(p.direction,Math.max(1,e.renderer.camera.distance))),m=[f[0],f[1],f[2]],w=c==="box"?{size:[.5,.5,.5]}:c==="sphere"?{radius:.5}:{radius:.4,height:.5};await e.props.onAddPrimitive(c,m,w),Nt("select"),yt("object");const x=be().filter(_=>_.kind===c),y=x[x.length-1];y&&de({nodeId:y.id,vertexIndex:null})},handlePlaceSketch:async(c,d)=>{const h=e.pickView(),u=wt(c,d,h),p=Xn();let f;if(p==="face"){const x=On(be(),u);if(!x){J({kind:"error",text:"On face: click a solid's surface."});return}f={origin:x.point,normal:x.normal}}else f=ss(p,u,e.renderer.camera.distance);await e.props.onAddSketch(f.origin);let m=mn(),w=m[m.length-1];f.normal&&w?.line!=null&&(await e.props.onSetValue(w.line,"PolygonProfile","planeNormal",f.normal),m=mn(),w=m[m.length-1]),Nt("select"),yt("object"),w&&de({nodeId:w.id,vertexIndex:null})},resolveFace:(c,d)=>{const h=wt(c,d,e.pickView()),u=On(be(),h);return u?sa(be(),u.point,u.normal):null},handleSketchOnFace:async(c,d)=>{const h=wt(c,d,e.pickView()),u=On(be(),h);if(!u){J({kind:"error",text:"Sketch on face: click a solid's surface."});return}const p=sa(be(),u.point,u.normal);if(p&&!p.face.usable){J({kind:"error",text:"That face belongs to a feature with no name in the source; assign it to a variable first."});return}const f={faceId:p?p.face.id:null,nodeId:p?p.nodeId:u.nodeId,near:u.point},m=Ie(),w=m?ae(m.nodeId):null,x=kt()==="sketch"&&w?.kind==="profile"&&w.line!==null?w.line:null;Yi(null),Nt("select"),yt("object"),await e.props.onSketchOnFace(f,x)},handleVertexConstraint:async(c,d,h)=>{const u=At[c],p=xn(Pe(),d,h,e.pickView());if(!p){J({kind:"error",text:`${u}: click a sketch point.`});return}const f=ae(p.nodeId);if(!f?.editable||f.line===null){J({kind:"error",text:"That sketch cannot be edited from source."});return}const m=e.pendingConstraint();if(m?.kind!==c){e.setPendingConstraint({kind:c,first:{nodeId:p.nodeId,vertexIndex:p.vertexIndex}}),de({nodeId:p.nodeId,vertexIndex:p.vertexIndex}),J({kind:"",text:`${u}: choose the second point`});return}const w=m.first;if(w.nodeId!==p.nodeId||w.vertexIndex===p.vertexIndex){J({kind:"error",text:"Choose a different point in the same sketch."});return}const x=[w.vertexIndex,p.vertexIndex];let y;if(c==="distance"){const _=f.vertices[w.vertexIndex].uv,$=f.vertices[p.vertexIndex].uv;y=Math.hypot($[0]-_[0],$[1]-_[1])}e.setPendingConstraint(null),await e.props.onAddConstraint(f.line,c,x,y),Nt("select"),de({nodeId:p.nodeId,vertexIndex:p.vertexIndex})},handleEdgeConstraint:async(c,d,h)=>{const u=At[c],p=ji(Pe(),d,h,e.pickView());if(!p){J({kind:"error",text:`${u}: click a sketch edge.`});return}const f=ae(p.nodeId);if(!f?.editable||f.line===null){J({kind:"error",text:"That sketch cannot be edited from source."});return}const[m,w]=yu(p.insertIndex,f.vertices.length),x=e.pendingConstraint();if(x?.kind!==c){e.setPendingConstraint({kind:c,first:{nodeId:p.nodeId,start:m,end:w}}),de({nodeId:p.nodeId,vertexIndex:null}),J({kind:"",text:`${u}: choose the second edge`});return}const y=x.first;if(y.nodeId!==p.nodeId||y.start===m){J({kind:"error",text:"Choose a different edge in the same sketch."});return}const _=[y.start,y.end,m,w];e.setPendingConstraint(null),await e.props.onAddConstraint(f.line,c,_),Nt("select"),de({nodeId:p.nodeId,vertexIndex:null})},handleLoftPick:async(c,d)=>{const h=ai();if(!h)return;const u=kn(Pe(),c,d,e.pickView(),10,!0),p=u&&ae(u.nodeId),f=Ku(h,p);if(f){J({kind:"error",text:f});return}gs(null),de({nodeId:p.id,vertexIndex:null}),await e.props.onAddLoft(h.line,p.line)},handleAddVertex:async(c,d)=>{const h=e.pickView(),u=mn().filter(x=>x.editable);if(u.length===0){J({kind:"error",text:"No editable sketch in this scene."});return}const p=ji(mn(),c,d,h),f=(p&&ae(p.nodeId))??(Ie()&&ae(Ie().nodeId))??u[0];if(!f.editable){J({kind:"error",text:"That sketch is not editable from the viewer."});return}const m=e.planePointAt(f.id,c,d);if(!m||f.line===null){J({kind:"error",text:"Click nearer the sketch plane to place a vertex."});return}const w=p&&p.nodeId===f.id?p.insertIndex:Zo(f,c,d,h);await e.props.onPatch("insert_vertex",f.line,w,m),de({nodeId:f.id,vertexIndex:Math.min(w,f.vertices.length)})}}}var Ju=S("<p class=viewer-hint data-testid=viewer-hint>");function Qu(e){return e.derived?`Sketch plane of ${e.name} · derived from a face (SketchPlane.${e.reference??"on"}): follows its parent, not draggable — edit the reference in the code`:e.editable?`Sketch plane of ${e.name} · click to select · drag the gizmo to move its origin`:`Sketch plane of ${e.name} · not editable from the viewer`}function eh(){const e=Ie(),t=e?ae(e.nodeId):null,n=kt()==="sketch"&&t?.kind==="profile"&&t.line!==null?`re-plants ${t.name??"the selected sketch"}`:"starts a new sketch",i=Mn();return i?i.face.usable?`Sketch on face: click ${oa(i.face)} — ${n} there · Esc to cancel`:`Sketch on face: ${oa(i.face)} has no name in the source — assign its feature to a variable first`:`Sketch on face: hover a flat face · a curved surface ${n} on a tangent plane · Esc to cancel`}function th(e){const t=e.name??"This point";return e.locked?`${t} · ${e.locked}`:e.state==="free"?`${t} · free parameter: the solid follows the drag`:`${t} · fixed value: the solid follows on release, after a recompile`}function nh(e){const t=()=>kt()==="simulate"?ri()?"Pick BC: click the mesh → sphere · Shift-drag → box · confirm in the builder":null:ai()?"Loft: click the second sketch in the viewport · Esc to cancel":re()==="sketch"?Xn()==="face"?"Sketch: click a solid's face to place it there · the ghost shows where the plane lands · Esc to cancel":`Sketch: click to place on the ${Xn().toUpperCase()} plane · the ghost shows where it lands · Esc to cancel`:re()==="face"?eh():re()==="polygon"?"Point: click sketch edges to add vertices · Esc to finish":tn(re())?e.pendingConstraint?`${At[re()]}: choose a second point in the same sketch`:`${At[re()]}: choose the first sketch point`:Ln(re())?e.pendingConstraint?`${At[re()]}: choose a second edge in the same sketch`:`${At[re()]}: choose the first sketch edge`:re()!=="select"?`Click to place a ${re()} · Esc to cancel`:e.handle?th(e.handle):e.plane?Qu(e.plane):null;return b(G,{get when(){return t()},children:n=>(()=>{var i=Ju();return g(i,n),i})()})}function zs(e){if(!e||!e.available||e.error)return null;const{positions:t,triangles:n,triangle_face:i,faces:r,edges:a,vertices:s,diagnostics:o}=e;return!t||!n||!i||!r||!a||!s||!o?null:{positions:t,triangles:n,triangleFace:i,cornerNormals:e.corner_normals&&e.corner_normals.length===n.length*3?e.corner_normals:null,faces:r,edges:a,vertices:s,diagnostics:o}}function ih(e,t=Vi.length){const n=new Map,i=new Map;for(const l of e.faces)n.set(l.id,new Set),i.set(l.id,new Set);const r=(l,c,d)=>{c===d||!l.has(c)||!l.has(d)||(l.get(c).add(d),l.get(d).add(c))};for(const l of e.edges)r(n,l.faces[0],l.faces[1]);for(const l of e.vertices)for(let c=0;c<l.faces.length;c++)for(let d=c+1;d<l.faces.length;d++)r(i,l.faces[c],l.faces[d]);const a=[...n.keys()].sort((l,c)=>n.get(c).size-n.get(l).size||l-c),s=new Map,o=new Array(t).fill(0);for(const l of a){let c=0,d=1/0;for(let h=0;h<t;h++){let u=0;for(const m of n.get(l))s.get(m)===h&&(u+=1);let p=0;for(const m of i.get(l))s.get(m)===h&&(p+=1);const f=u*1e6+p*1e3+o[h];f<d&&(d=f,c=h)}s.set(l,c),o[c]+=1}return s}function rh(e){return Vi[(e??0)%Vi.length]}function sr(e){const t=Un[e];if(t)return{rgb:le(se[t.token]),label:t.label,known:!0};const n=Un.other;return{rgb:le(se[n.token]),label:e,known:!1}}function ah(e){const t=new Map;for(const r of e.edges)t.set(r.kind,(t.get(r.kind)??0)+1);const n=Object.keys(Un).filter(r=>t.has(r)),i=[...t.keys()].filter(r=>!(r in Un)).sort();return[...n,...i].map(r=>{const a=sr(r);return{kind:r,label:a.label,rgb:a.rgb,count:t.get(r)}})}function Ns(e,t){return e.off_surface>t.diagnostics.off_surface_tolerance}const sh=.45,oh=.18,lh=.3;function ch(e){const t=le(se.ink),n=new Set,i=e.positions.length/3,r=[];for(let a=0;a<e.triangles.length;a+=3)for(let s=0;s<3;s++){const o=e.triangles[a+s],l=e.triangles[a+(s+1)%3];if(o===l)continue;const c=Math.min(o,l)*i+Math.max(o,l);n.has(c)||(n.add(c),r.push(e.positions[o*3],e.positions[o*3+1],e.positions[o*3+2],e.positions[l*3],e.positions[l*3+1],e.positions[l*3+2],t[0],t[1],t[2],lh))}return r}const dh=(e,t)=>[e[0]+(1-e[0])*t,e[1]+(1-e[1])*t,e[2]+(1-e[2])*t];function uh(e,t,n){const i=e.triangleFace.length,r=new Float32Array(i*3*10),a=e.cornerNormals,s=n?.kind==="face"?n.id:null;let o=0;for(let l=0;l<i;l++){const c=e.triangleFace[l],d=rh(t.get(c)),h=c===s?dh(d,sh):d;for(let u=0;u<3;u++){const p=e.triangles[l*3+u]*3;r[o++]=e.positions[p],r[o++]=e.positions[p+1],r[o++]=e.positions[p+2],r[o++]=h[0],r[o++]=h[1],r[o++]=h[2],r[o++]=1;const f=(l*3+u)*3;r[o++]=a?a[f]:0,r[o++]=a?a[f+1]:0,r[o++]=a?a[f+2]:0}}return r}function hh(e,t,n){if(n.hiddenKinds.has(e.kind))return null;const i=n.hover?.kind==="edge"&&n.hover.id===e.id;if(n.offSurfaceOnly){if(Ns(e,t))return[...le(se[Ka]),1];const a=sr(e.kind).rgb;return[a[0],a[1],a[2],i?1:oh]}const r=sr(e.kind).rgb;return[r[0],r[1],r[2],1]}function fh(e,t){const n=t.showTriangles?ch(e):[],i=(a,s)=>{const o=a.points,l=Math.floor(o.length/3),c=a.closed?l:l-1;for(let d=0;d<c;d++){const h=d*3,u=(d+1)%l*3;n.push(o[h],o[h+1],o[h+2],o[u],o[u+1],o[u+2],s[0],s[1],s[2],s[3])}};let r=null;for(const a of e.edges){const s=hh(a,e,t);s&&(t.hover?.kind==="edge"&&t.hover.id===a.id&&(r=a),i(a,s))}return r&&i(r,[...le(se.ink),1]),new Float32Array(n)}function ph(e,t){const n=new Float32Array(e.vertices.length*9),i=le(se.ink);return e.vertices.forEach((r,a)=>{const s=t?.kind==="vertex"&&t.id===r.id;n.set([...r.point,i[0],i[1],i[2],1,s?-.6:-1.5,2],a*9)}),n}const ca={vertex:8,edge:6};function mh(e,t,n,i,r=new Set){const a=gh(e,t,n,i),s=a?a.depth*1.03+1e-6:1/0;let o=null;for(const c of e.vertices){const d=ye(c.point,i);if(!d.visible||d.viewDepth>s)continue;const h=Math.hypot(d.x-t,d.y-n);h<=ca.vertex&&(!o||h<o.distance)&&(o={id:c.id,distance:h})}if(o)return{kind:"vertex",id:o.id};let l=null;for(const c of e.edges){if(r.has(c.kind))continue;const d=Math.floor(c.points.length/3);let h=null;const u=f=>ye([c.points[f*3],c.points[f*3+1],c.points[f*3+2]],i),p=d?u(0):null;h=p;for(let f=1;f<=(c.closed?d:d-1);f++){const m=f===d?p:u(f);if(h?.visible&&m.visible&&Math.min(h.viewDepth,m.viewDepth)<=s){const w=hn(t,n,h.x,h.y,m.x,m.y);w<=ca.edge&&(!l||w<l.distance)&&(l={id:c.id,distance:w})}h=m}}return l?{kind:"edge",id:l.id}:a?{kind:"face",id:a.id}:null}function gh(e,t,n,i){const r=e.positions.length/3,a=new Float64Array(r),s=new Float64Array(r),o=new Float64Array(r),l=new Uint8Array(r);for(let d=0;d<r;d++){const h=ye([e.positions[d*3],e.positions[d*3+1],e.positions[d*3+2]],i);a[d]=h.x,s[d]=h.y,o[d]=h.viewDepth,l[d]=h.visible?1:0}let c=null;for(let d=0;d<e.triangleFace.length;d++){const h=e.triangles[d*3],u=e.triangles[d*3+1],p=e.triangles[d*3+2];if(!l[h]||!l[u]||!l[p])continue;const f=(s[u]-s[p])*(a[h]-a[p])+(a[p]-a[u])*(s[h]-s[p]);if(Math.abs(f)<1e-12)continue;const m=((s[u]-s[p])*(t-a[p])+(a[p]-a[u])*(n-s[p]))/f,w=((s[p]-s[h])*(t-a[p])+(a[h]-a[p])*(n-s[p]))/f,x=1-m-w;if(m<0||w<0||x<0)continue;const y=m*o[h]+w*o[u]+x*o[p];(!c||y<c.depth)&&(c={id:e.triangleFace[d],depth:y})}return c}var vh=S("<dt>"),bh=S("<dd>"),Ir=S("<span>"),yh=S("<small>"),wh=S(`<div class=brep-legend data-testid=brep-legend><label class="brep-legend-row brep-off-surface"><input type=checkbox data-testid=brep-off-surface><span class=brep-swatch></span><span>Highlight off-surface</span><small data-testid=brep-off-surface-count></small></label><label class=brep-legend-row title="Outline every triangle of the extraction's tessellation, to judge how a face was triangulated."><input type=checkbox data-testid=brep-triangles><span class=brep-swatch style=opacity:0.3></span><span>Triangles</span><small data-testid=brep-triangle-count>`),_h=S("<label class=brep-legend-row><input type=checkbox><span class=brep-swatch></span><span></span><small>"),xh=S("<small>Hover a face, edge or vertex"),da=S("<small> (<!>)"),kh=S("<small> · leaf #"),Sh=S("<small> · patches "),$h=S("<small> · residual "),Ph=S('<div class=brep-inspector data-testid=brep-inspector><dl class="title-block brep-card">');const or=e=>e==null||!Number.isFinite(e)?"—":e===0?"0":Math.abs(e)<.001?e.toExponential(1):e.toFixed(4);function ve(e){return[(()=>{var t=vh();return g(t,()=>e.label),t})(),(()=>{var t=bh();return g(t,()=>e.children),D(()=>P(t,"data-testid",e.testId)),t})()]}const Eh=e=>Object.entries(e).sort((t,n)=>n[1]-t[1]||t[0].localeCompare(n[0])).map(([t,n])=>`${n} ${t}`).join(" · ");function Ih(e){const t=()=>e.model.diagnostics;return[b(ve,{label:"Path",testId:"brep-path",get children(){return X(()=>!!t().exact)()?"Exact · analytic":`Fallback · ${t().source.replace("_"," ")}`}}),b(G,{get when(){return X(()=>!t().exact)()&&t().fallback_reason},get children(){return b(ve,{label:"Why",testId:"brep-fallback-reason",get children(){var n=Ir();return g(n,()=>t().fallback_reason),D(()=>P(n,"title",t().fallback_reason??void 0)),n}})}}),b(ve,{label:"Faces",testId:"brep-faces",get children(){return[X(()=>t().faces)," ",(()=>{var n=yh();return g(n,()=>Eh(t().surfaces)),n})()]}}),b(ve,{label:"Edges",testId:"brep-edges",get children(){return t().edges}}),b(ve,{label:"Vertices",testId:"brep-vertices",get children(){return t().vertices}}),b(G,{get when(){return t().feature_curves},get children(){return b(ve,{label:"Curves",get children(){return t().feature_curves}})}})]}function Ah(e){const t=Q(()=>ah(e.model)),n=(r,a)=>{const s=new Set(Yn());a?s.delete(r):s.add(r),Xl(s)},i=()=>e.model.diagnostics;return(()=>{var r=wh(),a=r.firstChild,s=a.firstChild,o=s.nextSibling,l=o.nextSibling,c=l.nextSibling,d=a.nextSibling,h=d.firstChild,u=h.nextSibling,p=u.nextSibling,f=p.nextSibling;return g(r,b(Ae,{get each(){return t()},children:m=>(()=>{var w=_h(),x=w.firstChild,y=x.nextSibling,_=y.nextSibling,$=_.nextSibling;return x.addEventListener("change",E=>n(m.kind,E.currentTarget.checked)),g(_,()=>m.label),g($,()=>m.count),D(E=>{var I=`${m.count} ${m.kind} edges`,A=`brep-kind-${m.kind}`,B=Bs(m.rgb);return I!==E.e&&P(w,"title",E.e=I),A!==E.t&&P(x,"data-testid",E.t=A),B!==E.a&&Oe(y,"background",E.a=B),E},{e:void 0,t:void 0,a:void 0}),D(()=>x.checked=!Yn().has(m.kind)),w})()}),a),s.addEventListener("change",m=>Kl(m.currentTarget.checked)),g(c,()=>i().off_surface_edges),h.addEventListener("change",m=>Zl(m.currentTarget.checked)),g(f,()=>e.model.triangleFace.length),D(m=>{var w=`Edges whose points lie more than ${or(i().off_surface_tolerance)} off the solid (|f| of the scene): a curve under a smooth blend, drawn where the blend has rounded the corner away.`,x=se[Ka],y=se.ink;return w!==m.e&&P(a,"title",m.e=w),x!==m.t&&Oe(o,"background",m.t=x),y!==m.a&&Oe(u,"background",m.a=y),m},{e:void 0,t:void 0,a:void 0}),D(()=>s.checked=ps()),D(()=>h.checked=ms()),r})()}function Mh(e){const t=Q(()=>{const n=Wi();if(!n)return null;if(n.kind==="face"){const r=e.model.faces.find(a=>a.id===n.id);return r?{kind:"face",face:r}:null}if(n.kind==="edge"){const r=e.model.edges.find(a=>a.id===n.id);return r?{kind:"edge",edge:r}:null}const i=e.model.vertices.find(r=>r.id===n.id);return i?{kind:"vertex",vertex:i}:null});return b(G,{get when(){return t()},keyed:!0,get fallback(){return b(ve,{label:"Pointer",get children(){return xh()}})},children:n=>{if(n.kind==="face"){const r=n.face;return[b(ve,{label:"Face",testId:"brep-readout",get children(){return["#",X(()=>r.id)," · ",X(()=>r.surface),b(G,{get when(){return X(()=>r.surface==="faceted")()&&r.kind!=="freeform"},get children(){var a=da(),s=a.firstChild,o=s.nextSibling;return o.nextSibling,g(a,()=>r.kind,o),a}})]}}),b(ve,{label:"Patch",get children(){return[X(()=>X(()=>r.patch<0)()?"none (blend)":`#${r.patch}`),(()=>{var a=kh();return a.firstChild,g(a,()=>r.leaf,null),g(a,(()=>{var s=X(()=>!!r.leaf_type);return()=>s()?` ${r.leaf_type}`:""})(),null),a})()]}})]}if(n.kind==="edge"){const r=n.edge;return[b(ve,{label:"Edge",testId:"brep-readout",get children(){return[X(()=>X(()=>r.id<0)()?"curve":`#${r.id}`)," · ",X(()=>r.kind),b(G,{get when(){return r.state!==r.kind},get children(){var a=da(),s=a.firstChild,o=s.nextSibling;return o.nextSibling,g(a,()=>r.state,o),a}})]}}),b(ve,{label:"Faces",get children(){return[X(()=>r.faces.map(a=>a<0?"—":`#${a}`).join(" | ")),(()=>{var a=Sh();return a.firstChild,g(a,()=>r.patches.map(s=>s<0?"blend":`#${s}`).join(" | "),null),a})()]}}),b(ve,{label:"Off solid",get children(){return[(()=>{var a=Ir();return g(a,()=>or(r.off_surface)),D(()=>a.classList.toggle("brep-alert",!!Ns(r,e.model))),a})(),(()=>{var a=$h();return a.firstChild,g(a,()=>or(r.residual),null),a})()]}})]}const i=n.vertex;return[b(ve,{label:"Vertex",testId:"brep-readout",get children(){return["#",X(()=>i.id)," · ",X(()=>i.kind)]}}),b(ve,{label:"At",get children(){return i.point.map(r=>r.toFixed(4)).join(", ")}}),b(ve,{label:"Faces",get children(){return i.faces.map(r=>`#${r}`).join(" ")}})]}})}function Ch(e){const t=Q(()=>zs(fs()));return b(G,{get when(){return e.show},get children(){var n=Ph(),i=n.firstChild;return g(i,b(ve,{label:"B-rep",testId:"brep-status",get children(){return X(()=>In()==="loading")()?"Deriving…":X(()=>In()==="error")()?"Extraction failed":X(()=>In()==="unavailable")()?"Dual contouring only · public tier":t()?"Derived":"Waiting for a compile"}}),null),g(i,b(G,{get when(){return An()},get children(){return b(ve,{label:"Note",testId:"brep-message",get children(){var r=Ir();return g(r,An),D(()=>P(r,"title",An()??void 0)),r}})}}),null),g(i,b(G,{get when(){return t()},children:r=>[b(Ih,{get model(){return r()}}),b(Mh,{get model(){return r()}})]}),null),g(n,b(G,{get when(){return t()},children:r=>b(Ah,{get model(){return r()}})}),null),n}})}function Th(e){if(!Number.isFinite(e))return"–";const t=Math.abs(e);return t!==0&&(t>=1e4||t<.001)?e.toExponential(2):e.toFixed(3).replace(/\.?0+$/,"")||"0"}const Oh={axis:0,fraction:1,enabled:!1};function Lh(e){const t=[1/0,1/0,1/0],n=[-1/0,-1/0,-1/0];for(let i=0;i+2<e.length;i+=3)for(let r=0;r<3;r++){const a=e[i+r];a<t[r]&&(t[r]=a),a>n[r]&&(n[r]=a)}return Number.isFinite(t[0])?{min:t,max:n}:{min:[0,0,0],max:[0,0,0]}}function Dh(e,t){const n=[0,0,0];n[e.axis]=1;const i=t.min[e.axis],r=t.max[e.axis],a=Math.max(1e-6,(r-i)*1e-4),s=Math.min(1,Math.max(0,e.fraction)),o=i-a+s*(r-i+2*a);return{normal:n,offset:o}}function Rh(e,t,n){const i=new Map,r=Math.min(t+n,e.length);for(let s=t;s+2<r;s+=3){const o=[e[s],e[s+1],e[s+2]];for(let l=0;l<3;l++){const c=o[l],d=o[(l+1)%3],h=c<d?`${c},${d}`:`${d},${c}`,u=i.get(h);u?u[2]+=1:i.set(h,[c,d,1])}}const a=[];for(const[s,o,l]of i.values())l===1&&a.push(s,o);return new Uint32Array(a)}var Bh=S("<div class=bc-pick-rect data-testid=bc-pick-rect>"),Fh=S("<div class=sim-probe data-testid=sim-probe><b></b><span></span><small>[<!>]"),zh=S("<div class=viewer-error><p></p><button type=button>Dismiss");function Nh(e){return[b(G,{get when(){return e.pickRect},children:t=>(()=>{var n=Bh();return D(i=>{var r=`${t().left}px`,a=`${t().top}px`,s=`${t().width}px`,o=`${t().height}px`;return r!==i.e&&Oe(n,"left",i.e=r),a!==i.t&&Oe(n,"top",i.t=a),s!==i.a&&Oe(n,"width",i.a=s),o!==i.o&&Oe(n,"height",i.o=o),i},{e:void 0,t:void 0,a:void 0,o:void 0}),n})()}),b(G,{get when(){return X(()=>kt()==="simulate")()&&hs()},children:t=>(()=>{var n=Fh(),i=n.firstChild,r=i.nextSibling,a=r.nextSibling,s=a.firstChild,o=s.nextSibling;return o.nextSibling,g(i,()=>Th(t().value)),g(r,()=>t().label),g(a,()=>t().world.map(l=>l.toFixed(3)).join(", "),o),D(l=>{var c=`${t().x}px`,d=`${t().y}px`;return c!==l.e&&Oe(n,"left",l.e=c),d!==l.t&&Oe(n,"top",l.t=d),l},{e:void 0,t:void 0}),n})()}),X(()=>X(()=>!!dn())()&&(()=>{var t=zh(),n=t.firstChild,i=n.nextSibling;return g(n,dn),Wa(i,"click",tc,!0),t})())]}Ce(["click"]);const ua="cadjoint:dock-rebuilt";var Gh=S('<section class="pane viewer-pane"><canvas class=viewer-canvas data-testid=viewer-canvas>');function jh(e){let t,n={kind:"none"};const[i,r]=T(null);let a=!1;const[s,o]=T(!1),[l,c]=T(0),[d,h]=T(null),u=e.renderer,p=v=>{const k=t.getBoundingClientRect(),C=t.width/Math.max(k.width,1),M=t.height/Math.max(k.height,1);return[(v.clientX-k.left)*C,(v.clientY-k.top)*M]},f=()=>u.view,m=()=>c(v=>v+1),w=(v,k,C)=>{const M=ae(v);if(!M?.plane)return null;const{origin:W,normal:F,u:O,v:Y}=M.plane,z=Xa(wt(k,C,f()),W,F);return z?Ko(z,W,O,Y):null},x=Zu({renderer:u,pickView:f,planePointAt:w,pendingConstraint:i,setPendingConstraint:r,props:e}),y=Hu({renderer:u,pickView:f,props:e}),_=Uu({canvas:()=>t,renderer:u,pickView:f}),$=Q(()=>{l();const{distance:v,yaw:k,pitch:C}=u.camera;return{distance:v,yaw:k,pitch:C,projection:e.display.projection}}),E=Q(()=>(l(),wu(f(),t?.clientWidth??0,t?.clientHeight??0,Pe(),zl()))),I=Q(()=>{const v=Yr();return v?Ol(v.origin,v.normal):null}),A=Q(()=>Bl(cc(),{sketching:kt()==="sketch",selectedNodeId:Ie()?.nodeId??null,hoveredNodeId:nt()?.nodeId??null,showOverlays:e.display.showOverlays,showSketches:e.display.showSketches,showAllPlanes:e.display.showAllPlanes===!0})),B=Q(()=>(l(),Pu(f(),t?.clientWidth??0,t?.clientHeight??0,A(),I()))),N=Q(()=>{if(!e.display.showOverlays)return null;const v=nt();if(!v||v.part!=="plane")return null;const k=ae(v.nodeId);return k?.plane?{name:k.name??k.id,derived:k.plane.reference!=null,editable:k.editable&&k.transform!==null,reference:k.plane.reference?.constructor??null}:null}),L=(v,k)=>{const C=wt(v,k,f()),M=Xn();if(M==="face"){const F=On(be(),C);fi(F?{origin:F.point,normal:F.normal}:null);return}const W=ss(M,C,u.camera.distance);fi({origin:W.origin,normal:W.normal??[0,0,1]})},H=Q(()=>{if(!e.display.showOverlays||!e.display.showSketches)return null;const v=nt();if(!v||v.vertexIndex===null)return null;const k=ae(v.nodeId)?.vertices[v.vertexIndex];return k?{name:k.name,state:hu(k,u.parameterProgram),locked:e.lockedHandle?.(v.nodeId,v.vertexIndex)??null}:null}),Z=(v,k)=>{if(U(v,k),a){t.style.cursor="move";return}if(re()==="face"){const Y=x.resolveFace(v,k);Y?.face.id!==Mn()?.face.id&&Yi(Y),t.style.cursor=Y?"crosshair":"cell",it(null),u.gizmoAxis=null;return}if(tn(re())){const Y=xn(Pe(),v,k,f()),z=Y?{nodeId:Y.nodeId,vertexIndex:Y.vertexIndex}:null;it(z),t.style.cursor=Y?"crosshair":"default",u.gizmoAxis=null;return}if(Ln(re())){const Y=ji(Pe(),v,k,f());it(null),t.style.cursor=Y?"crosshair":"default",u.gizmoAxis=null;return}if(re()!=="select"){re()==="sketch"&&L(v,k),t.style.cursor="crosshair",it(null),u.gizmoAxis=null;return}const C=f();if(y.hoverAxis(v,k,C)){t.style.cursor="grab";return}if(It()==="vertex"){const Y=xn(Pe(),v,k,C),z=Y?{nodeId:Y.nodeId,vertexIndex:Y.vertexIndex}:null,te=nt();(z?.nodeId!==te?.nodeId||z?.vertexIndex!==te?.vertexIndex)&&it(z),t.style.cursor=Y?"grab":"default";return}const M=kn(Pe(),v,k,C,10,!0),W=!M&&e.display.showOverlays?Gr(A(),v,k,C):null,F=M?{nodeId:M.nodeId,vertexIndex:null}:W?{nodeId:W.nodeId,vertexIndex:null,part:"plane"}:null,O=nt();(F?.nodeId!==O?.nodeId||F?.vertexIndex!==O?.vertexIndex||F?.part!==O?.part)&&it(F),t.style.cursor=M||W?"pointer":"default"},ie=v=>{t.setPointerCapture(v.pointerId);const[k,C]=p(v);if(_.simInteractive()&&v.button===0&&!a){if(ri()&&v.shiftKey){n={kind:"bcrect",x0:k,y0:C,x1:k,y1:C},h(_.rectFromGesture(n));return}if(!v.shiftKey){n={kind:"simtap",x:k,y:C,clientX:v.clientX,clientY:v.clientY};return}}if(re()==="sketch"&&v.button===0){x.handlePlaceSketch(k,C);return}if(re()==="face"&&v.button===0){x.handleSketchOnFace(k,C);return}if(ai()&&v.button===0){x.handleLoftPick(k,C);return}const M=re();if(tn(M)&&v.button===0){x.handleVertexConstraint(M,k,C);return}if(Ln(M)&&v.button===0){x.handleEdgeConstraint(M,k,C);return}if(v.button===0&&(re()==="box"||re()==="sphere"||re()==="cylinder")){x.handlePlacePrimitive(re(),k,C);return}if(re()==="polygon"&&v.button===0){x.handleAddVertex(k,C);return}if(v.button===0){const F=y.begin(k,C),O=F&&ar(F,ae(F.nodeId),F)?.argument,Y=O?e.lockedHandle?.(F.nodeId,O):null;if(F!==null&&Y){const z=ae(F.nodeId)?.name??"This object";J({kind:"",text:`${z} · ${O} · ${Y}`}),n={kind:"none"};return}if(F!==null){n=F,u.gizmoAxis=F.axis,u.interacting=!0,u.invalidate();return}}const W=v.button===0&&It()==="vertex"?xn(Pe(),k,C,f()):null;if(W){const F=ae(W.nodeId);de({nodeId:W.nodeId,vertexIndex:W.vertexIndex});const O=e.lockedHandle?.(W.nodeId,W.vertexIndex)??null;if(O){const Y=F?.vertices[W.vertexIndex]?.name??"This point";J({kind:"",text:`${Y} · ${O}`})}F?.editable&&!O?(n={kind:"drag",nodeId:W.nodeId,vertexIndex:W.vertexIndex,moved:!1},u.interacting=!0,u.invalidate()):n={kind:"none"};return}if(v.button===0&&It()==="object"){const F=kn(Pe(),k,C,f(),10,!0);if(F){ae(F.nodeId)?.kind==="profile"&&Kn("translate"),de({nodeId:F.nodeId,vertexIndex:null}),n={kind:"none"};return}const O=e.display.showOverlays?Gr(A(),k,C,f()):null;if(O){Kn("translate"),de({nodeId:O.nodeId,vertexIndex:null,part:"plane"});const Y=ae(O.nodeId);Y?.plane?.reference&&J({kind:"",text:`${Y.name??"This sketch"}'s plane is derived from a face (SketchPlane.${Y.plane.reference.constructor}); it follows its parent and cannot be dragged — edit the reference in the code.`}),n={kind:"none"};return}de(null)}else v.button===0&&de(null);n=v.button===2||v.shiftKey||a?{kind:"pan",x:v.clientX,y:v.clientY}:{kind:"orbit",x:v.clientX,y:v.clientY},u.interacting=!0},ce=v=>{const[k,C]=p(v);if(v.buttons===0&&Ue()){Ne(),Z(k,C);return}if(n.kind==="none"){Z(k,C);return}if(n.kind==="simtap"){Math.hypot(v.clientX-n.clientX,v.clientY-n.clientY)>4&&(Wn(null),n={kind:"orbit",x:v.clientX,y:v.clientY},u.interacting=!0);return}if(n.kind==="bcrect"){n.x1=k,n.y1=C,h(_.rectFromGesture(n));return}if(n.kind==="gizmo"){t.style.cursor="grabbing",y.update(n,k,C);return}if(n.kind==="drag"){t.style.cursor="grabbing";const M=w(n.nodeId,k,C);if(!M)return;n.moved=!0,gi({nodeId:n.nodeId,vertexIndex:n.vertexIndex,xy:M});const W=ae(n.nodeId)?.vertices[n.vertexIndex],F=W?Ts(W.binding?[W.binding]:null,M,u.parameterProgram):null;F&&(u.setParameterOverrides(F),e.onLiveParameters?.(F));return}if(n.kind==="orbit"){t.style.cursor="grabbing",u.camera=Hd(u.camera,v.clientX-n.x,v.clientY-n.y),n.x=v.clientX,n.y=v.clientY,xr({yaw:u.camera.yaw,pitch:u.camera.pitch}),m(),u.invalidate();return}t.style.cursor="move",u.camera=Ud(u.camera,v.clientX-n.x,v.clientY-n.y),n.x=v.clientX,n.y=v.clientY,m(),u.invalidate()},Re=async v=>{t.hasPointerCapture(v.pointerId)&&t.releasePointerCapture(v.pointerId);const k=n;if(n={kind:"none"},u.interacting=!1,t.style.cursor="default",k.kind==="simtap"){_.handleSimTap(k.x,k.y);return}if(k.kind==="bcrect"){h(null);const C=qn();if(C){const M=du(C.payload.positions,k,f());M?us(M):J({kind:"error",text:"Box pick: drag the rectangle over the mesh."})}return}if(k.kind==="gizmo"){await y.commit(k);return}if(k.kind==="drag"){const C=Xi(),M=ae(k.nodeId);gi(null),k.moved&&C&&M?.line!=null?(await e.onPatch("set_vertex",M.line,k.vertexIndex,C.xy),u.dropStaleParameterOverrides()):k.moved&&u.setParameterOverrides(null);return}k.kind!=="none"&&u.invalidate()},ue=v=>{v.preventDefault(),u.camera=v.altKey?Yd(u.camera,v.deltaY):Wd(u.camera,v.deltaY),m(),u.invalidate()},ze=async v=>{v.preventDefault(),o(!1);const k=v.dataTransfer?.getData(ki)??v.dataTransfer?.getData("text/plain");if(!k)return;const[C,M]=p(v),W=kn(Pe(),C,M,f(),64,!0),F=W&&ae(W.nodeId);if(!F?.editable||F.line===null){J({kind:"error",text:"Drop closer to an editable object."});return}de({nodeId:F.id,vertexIndex:null}),J({kind:"",text:`Applying ${k}…`}),await e.onAssignMaterial(F.line,k)},Ue=()=>n.kind==="drag"||n.kind==="gizmo"||n.kind==="bcrect",qe=()=>Xi()!==null||_r()!==null||d()!==null,Ne=()=>{!Ue()&&!qe()||(n={kind:"none"},gi(null),Qt(null),h(null),u.gizmoAxis=null,u.interacting=!1,u.setParameterOverrides(null),u.setConstruction(Pe(),Ie(),nt(),A()),u.invalidate())},xe=Xu({props:e,clearPendingConstraint:()=>r(null),pendingConstraintActive:()=>i()!==null,gestureActive:qe,cancelGesture:Ne,setPanHeld:v=>{a=v,t.style.cursor=a?"move":"default"}});pr(()=>{u.init(t),u.resize();const v=new ResizeObserver(()=>{u.resize(),m(),u.invalidate()});v.observe(t);const k=()=>{u.reconfigure(),u.resize(),m(),u.invalidate()};window.addEventListener("keydown",xe.onKeyDown),window.addEventListener("keydown",xe.onPanKey),window.addEventListener("keyup",xe.onPanKey),window.addEventListener(ua,k),ft(()=>{v.disconnect(),window.removeEventListener("keydown",xe.onKeyDown),window.removeEventListener("keydown",xe.onPanKey),window.removeEventListener("keyup",xe.onPanKey),window.removeEventListener(ua,k),u.destroy()})}),Te(()=>{u.setConstruction(Pe(),Ie(),nt(),A())}),Te(()=>{const v=ds();u.setFocusGroup(v===null?new Set:mc(cs(),v))}),Te(()=>{u.setMeshEdges(Wl())});const et=Q(()=>zs(fs())),Wt=Q(()=>{const v=et();return v?ih(v):new Map});Te(()=>{const v=et();if(!e.display.showBrep||!v){u.setBrep(null);return}const k={hiddenKinds:Yn(),offSurfaceOnly:ps(),showTriangles:ms(),hover:Wi()};u.setBrep({faces:uh(v,Wt(),k.hover),edges:fh(v,k),vertices:ph(v,k.hover)})});let st=null,j=0;const R=(v,k)=>v===k||v!==null&&k!==null&&v.kind===k.kind&&v.id===k.id,U=(v,k)=>{!e.display.showBrep||!et()||(st=[v,k],!j&&(j=requestAnimationFrame(()=>{j=0;const C=et(),M=st;if(!C||!M)return;const W=mh(C,M[0],M[1],f(),Yn());R(W,Wi())||Yl(W)})))};return ft(()=>cancelAnimationFrame(j)),Te(()=>{re()!=="sketch"&&Yr()!==null&&fi(null)}),Te(()=>{u.setPlanePreview(e.display.showOverlays?I():null)}),Te(()=>{if(re()!=="face"){Mn()!==null&&Yi(null),u.setFaceHighlight(null);return}u.setFaceHighlight(Mn()?.face??null)}),Te(()=>{const v=Ie(),C=(v?ae(v.nodeId):null)?.plane?.normal,M=!C||Math.abs(C[2])>.999,F=kt()==="sketch"&&!M?Tn.offPlane:1;u.groundEmphasis!==F&&(u.groundEmphasis=F,u.invalidate())}),Te(()=>{Ki(),e.display.projection,m()}),Te(()=>{const v=re();if(!tn(v)&&!Ln(v)){r(null);return}if(i()&&i().kind!==v&&r(null),!tn(v))return;const k=Ie();!i()&&k?.vertexIndex!=null&&ae(k.nodeId)?.kind==="profile"&&(r({kind:v,first:{nodeId:k.nodeId,vertexIndex:k.vertexIndex}}),J({kind:"",text:`${At[v]}: choose the second point`}))}),(()=>{var v=Gh(),k=v.firstChild;g(v,()=>e.overlay,k),k.$$dblclick=()=>{u.camera={...u.camera,target:[0,0,0]},m(),u.invalidate()},k.$$contextmenu=M=>M.preventDefault(),k.addEventListener("drop",M=>void ze(M)),k.addEventListener("dragleave",()=>o(!1)),k.addEventListener("dragover",M=>{M.dataTransfer?.types.includes(ki)&&(M.preventDefault(),M.dataTransfer&&(M.dataTransfer.dropEffect="copy"))}),k.addEventListener("dragenter",M=>{M.dataTransfer?.types.includes(ki)&&o(!0)}),k.addEventListener("wheel",ue),k.addEventListener("pointercancel",M=>void Re(M)),k.$$pointerup=M=>void Re(M),k.$$pointermove=ce,k.$$pointerdown=ie;var C=t;return typeof C=="function"?on(C,k):t=k,g(v,b($u,{get show(){return X(()=>!!e.display.showOverlays)()&&e.display.showConstraints},get showDistance(){return e.display.showDistanceConstraints},get showFixed(){return e.display.showFixedConstraints},get showValues(){return e.display.showConstraintValues},get geometry(){return E()}}),null),g(v,b(Vu,{get show(){return e.display.showGraticule},get camera(){return $()},get sdfView(){return e.display.sdfView},get sdfAxis(){return e.display.sdfAxis},get sdfFraction(){return e.display.sdfFraction}}),null),g(v,b(Du,{get renderer(){return e.renderer},get show(){return e.display.showGraticule}}),null),g(v,b(Ch,{get show(){return e.display.showBrep===!0}}),null),g(v,b(Au,{get show(){return e.display.showOverlays},get geometry(){return B()}}),null),g(v,b(Nh,{get pickRect(){return X(()=>!!e.display.showOverlays)()?d():null}}),null),g(v,b(nh,{get pendingConstraint(){return i()},get handle(){return H()},get plane(){return N()}}),null),D(M=>{var W=!!e.display.showGraticule,F=e.display.sdfView!=="solid",O=nc(),Y=!!s();return W!==M.e&&v.classList.toggle("has-graticule",M.e=W),F!==M.t&&v.classList.toggle("sdf-data",M.t=F),O!==M.a&&P(v,"aria-busy",M.a=O),Y!==M.o&&k.classList.toggle("material-drop-active",M.o=Y),M},{e:void 0,t:void 0,a:void 0,o:void 0}),v})()}Ce(["pointerdown","pointermove","pointerup","contextmenu","dblclick"]);const Gs="cadjoint.render-presets.v5",Jn=[{id:"xray",name:"X-Ray",hint:"Model & inspect",pathTracing:!1,quality:Pn,display:{...jt}},{id:"studio",name:"Studio",hint:"Clean materials",pathTracing:!0,quality:Pn,display:{...jt,projection:"perspective",shadows:"soft",reflections:!0,flatShading:!1,xray:0,showGraticule:!1,showOverlays:!1,showSketches:!1,showConstraints:!1}},{id:"wire",name:"Wire",hint:"Construction only",pathTracing:!1,quality:Pn,display:{...jt,projection:"orthographic",shadows:"off",reflections:!1,flatShading:!0,hideSolid:!0,xray:0,showSketches:!0,showConstraints:!0}}],js=e=>({...e,display:{...e.display}});function Vh(){return Jn.map(js)}function Hh(e){if(!e||typeof e!="object")return!1;const t=e;return(t.projection==="perspective"||t.projection==="orthographic")&&(t.shadows==="off"||t.shadows==="hard"||t.shadows==="soft")&&typeof t.reflections=="boolean"&&typeof t.flatShading=="boolean"&&typeof t.hideSolid=="boolean"&&typeof t.xray=="number"&&Number.isFinite(t.xray)&&typeof t.showGraticule=="boolean"&&typeof t.showOverlays=="boolean"&&typeof t.showSketches=="boolean"&&typeof t.showMeshEdges=="boolean"&&typeof t.showMeshWireframe=="boolean"&&typeof t.showConstraints=="boolean"&&typeof t.showFixedConstraints=="boolean"&&typeof t.showDistanceConstraints=="boolean"&&typeof t.showConstraintValues=="boolean"&&(t.sdfView==="solid"||t.sdfView==="slice"||t.sdfView==="gradient")&&(t.sdfAxis===0||t.sdfAxis===1||t.sdfAxis===2)&&typeof t.sdfFraction=="number"&&Number.isFinite(t.sdfFraction)&&typeof t.isoOffset=="number"&&Number.isFinite(t.isoOffset)&&(t.marchSteps===null||typeof t.marchSteps=="number"&&Number.isFinite(t.marchSteps))&&typeof t.refineHit=="boolean"&&typeof t.cullBounds=="boolean"&&typeof t.section=="boolean"}function Uh(e=ii()){const t={presets:Vh(),activeId:"xray"};if(!e)return t;try{const n=JSON.parse(e.getItem(Gs)??"null");if(!n||!Array.isArray(n.presets))return t;const i=n.presets,r=Jn.map(s=>{const o=i.find(l=>l&&typeof l=="object"&&l.id===s.id);return!o||!Hh(o.display)||typeof o.pathTracing!="boolean"||typeof o.quality!="string"||!Ze[o.quality]?js(s):{...s,pathTracing:o.pathTracing,quality:o.quality,display:{...o.display}}}),a=Jn.some(s=>s.id===n.activeId)?n.activeId:"xray";return{presets:r,activeId:a}}catch{return t}}function Si(e,t=ii()){if(t)try{t.setItem(Gs,JSON.stringify({activeId:e.activeId,presets:e.presets.map(({id:n,pathTracing:i,quality:r,display:a})=>({id:n,pathTracing:i,quality:r,display:a}))}))}catch{}}function ha(e,t,n,i){return e.pathTracing===i&&e.quality===n&&Object.keys(jt).every(r=>e.display[r]===t[r])}var fa=S("<small>"),qh=S('<div class=sim-slice><input type=range min=0 max=1 step=0.005 aria-label="Slice position"data-testid=sdf-fraction>'),Wh=S("<div class=sim-legend-values><span> = </span><span>± "),Yh=S("<div class=sim-legend data-testid=sdf-legend><small></small><div class=sim-ramp></div><div class=sim-legend-values><span></span><span></span><span></span></div><small>"),Xh=S('<div class=sim-legend data-testid=sdf-normal-legend><small>n × 0.5 + 0.5, in world axes</small><div class="sim-ramp sdf-axis-ramp"></div><div class=sim-legend-values><span>+X red</span><span>+Y green</span><span>+Z blue'),Kh=S('<div class=sim-legend data-testid=sdf-depth-legend><small>Linear depth along the primary ray</small><div class="sim-ramp sdf-depth-ramp"></div><div class=sim-legend-values><span>near</span><span>far</span></div><small>Half a frame either side of the orbit target'),Zh=S("<small>Resolves the hit off the last two samples. Moves 0.1–2 % of the frame, for 0–30 % of one."),Jh=S("<small>Skips the leaves a ray is far from. Same image; off is 2.0–2.4× the frame."),Qh=S("<small>Progressive physically based rendering"),ef=S("<small>See construction through the solid"),tf=S("<small class=tier-reason>"),nf=S("<small>Floor plane, scale readout, and title block"),rf=S("<small>Sketches, handles, gizmo, constraints, BC preview"),af=S("<small>Edges and editable point handles"),sf=S("<small>Otherwise only the active sketch's, while sketching"),of=S("<small>Viewport dimensions and fixed badges"),lf=S('<div class=render-preset-editor data-testid=render-preset-editor><div class=render-preset-actions><span></span><button type=button data-testid=render-preset-save>Save</button><button type=button data-testid=render-preset-reset>Reset</button></div><section><h4>Shading</h4></section><section><h4>Shadows</h4></section><section data-testid=render-resolution><div class=render-preset-heading><h4>Resolution</h4><small>How well it is drawn</small></div><small class=pane-hint></small></section><section data-testid=render-march><div class=render-preset-heading><h4>Marching</h4><small>A cap and a refinement</small></div><div class=sim-slice><span class=pane-hint>Steps</span><input type=range step=16 aria-label="Sphere-tracing step budget"data-testid=march-steps></div><div class=sim-legend-values><span data-testid=march-steps-value> steps</span><button type=button class=render-sdf-zero data-testid=march-steps-tier></button></div><div class=sim-legend><small>A cap, not a cost: raising it is nearly free, lowering it erodes the silhouette.</small></div></section><section data-testid=render-diagnostics><div class=render-preset-heading><h4>Diagnostics</h4><small>Cost, not appearance</small></div></section><section><h4>Effects</h4></section><section><h4>Annotations</h4><div class=annotation-options>'),cf=S('<aside class=render-panel data-testid=render-panel><div class=render-settings><section class=render-preset-library><div class=render-preset-heading><h4>Render presets</h4><small>Click to activate</small></div><div class=render-preset-grid></div><button type=button data-testid=render-customize><span><b></b><small></small></span><i></i></button></section><section class=render-sdf data-testid=render-sdf><div class=render-preset-heading><h4>Distance field</h4><small>What the viewport draws</small></div><div class=sim-slice><span class=pane-hint>Offset</span><input type=range min=-0.4 max=0.4 step=0.005 aria-label="Isosurface offset"data-testid=sdf-offset></div><div class=sim-legend-values><span>f = </span><button type=button class=render-sdf-zero data-testid=sdf-offset-zero>f = 0'),df=S("<button type=button class=render-preset-card><span class=render-preset-preview aria-hidden=true></span><strong></strong><small>"),uf=S("<small>Faces, edges and vertices the extraction found");const hf={xray:Vc,studio:Hc,wire:Uc},ff=[{value:"off",label:"Off",title:"No shadow rays",testId:"shadows-off"},{value:"hard",label:"Hard",title:"One crisp, lifted shadow ray",testId:"shadows-hard"},{value:"soft",label:"Soft",title:"Penumbra from multiple samples",testId:"shadows-soft"}],pf=[{value:!1,label:"Full",testId:"shading-full"},{value:!0,label:"Flat",title:"Albedo only, no specular or environment",testId:"shading-flat"}],mf=Object.entries(Ze).map(([e,t])=>({value:e,label:t.label,testId:`quality-${e}`})),gf=[{value:"solid",label:"Solid",title:"The raymarched surface",testId:"sdf-solid"},{value:"slice",label:"Slice",title:"Signed distance on a plane through the scene",testId:"sdf-slice"},{value:"gradient",label:"∇f",title:"Gradient magnitude: where the field stops being a metric distance",testId:"sdf-gradient"},{value:"normal",label:"N",title:"World-space surface normals, n × 0.5 + 0.5",testId:"sdf-normal"},{value:"depth",label:"Z",title:"Linear camera depth across the framed volume",testId:"sdf-depth"}],vf=Wc("sdf-axis"),$i=e=>{const{value:t,unit:n}=As(e*Vt);return`${t} ${n}`},bf=[{key:"reflections",label:"Reflections",hint:"Environment reflections"},{key:"hideSolid",label:"Hide solid",hint:"Construction geometry only"},{key:"showMeshEdges",label:"Feature edges",hint:"Sharp creases, corners, and CSG seams",tier:"feature_edges"},{key:"showMeshWireframe",label:"Mesh wireframe",hint:"Full dual-contour quad wireframe"}],Pi=()=>ws("brep")||In()==="unavailable";function yf(e){const[t,n]=T(!1),i=Q(()=>e.presets.find(l=>l.id===e.selectedPreset)??e.presets[0]),r=()=>{const l=Ze[e.quality]??Ze.ultra,c=(l.pixelBudget/1e6).toFixed(2);return e.pathTracing?`${c} MP · ${l.bounces} bounces`:`${c} MP · all a tier costs`},a=()=>es(e.display,Ze[e.quality]??Ze.ultra),s=()=>{const l=Ze[e.quality]??Ze.ultra;return e.display.marchSteps===null?`${l.label} tier`:`Tier (${l.marchSteps})`},o=()=>ha(i(),e.display,e.quality,e.pathTracing);return(()=>{var l=cf(),c=l.firstChild,d=c.firstChild,h=d.firstChild,u=h.nextSibling,p=u.nextSibling,f=p.firstChild,m=f.firstChild,w=m.nextSibling,x=f.nextSibling,y=d.nextSibling,_=y.firstChild,$=_.nextSibling,E=$.firstChild,I=E.nextSibling,A=$.nextSibling,B=A.firstChild;B.firstChild;var N=B.nextSibling;return g(l,b(Sr,{kicker:"presets & display",title:"Render"}),c),g(u,b(Ae,{get each(){return e.presets},children:L=>{const H=()=>ha(L,e.display,e.quality,e.pathTracing);return(()=>{var Z=df(),ie=Z.firstChild,ce=ie.nextSibling,Re=ce.nextSibling;return Z.$$click=()=>e.onPresetActivate(L.id),g(ie,()=>(hf[L.id]??bs)({size:24})),g(ce,()=>L.name),g(Re,()=>L.hint),D(ue=>{var ze=!!H(),Ue=L.id===e.selectedPreset,qe=H(),Ne=L.id,xe=`render-preset-${L.id}`;return ze!==ue.e&&Z.classList.toggle("active",ue.e=ze),Ue!==ue.t&&Z.classList.toggle("selected",ue.t=Ue),qe!==ue.a&&P(Z,"aria-pressed",ue.a=qe),Ne!==ue.o&&P(Z,"data-preset",ue.o=Ne),xe!==ue.i&&P(Z,"data-testid",ue.i=xe),ue},{e:void 0,t:void 0,a:void 0,o:void 0,i:void 0}),Z})()}})),p.$$click=()=>n(!t()),g(m,()=>t()?"Editing":"Customize"),g(w,()=>i().name),g(x,()=>t()?"−":"+"),g(y,b(Xt,{options:gf,get value(){return e.display.sdfView},onSelect:L=>e.onChange({sdfView:L})}),$),g(y,b(ke,{get checked(){return e.display.section},onChange:L=>e.onChange({section:L}),testId:"toggle-section",get children(){return["Capped section",(()=>{var L=fa();return g(L,()=>En(e.display.sdfView)?"The field card covers the cut":"Closes the cut with a real face"),L})()]}}),$),g(y,b(G,{get when(){return En(e.display.sdfView)||e.display.section},get children(){return[(()=>{var L=qh(),H=L.firstChild;return g(L,b(Xt,{options:vf,get value(){return e.display.sdfAxis},onSelect:Z=>e.onChange({sdfAxis:Z})}),H),H.$$input=Z=>e.onChange({sdfFraction:Number(Z.currentTarget.value)}),D(()=>H.value=e.display.sdfFraction),L})(),(()=>{var L=Wh(),H=L.firstChild,Z=H.firstChild,ie=H.nextSibling;return ie.firstChild,g(H,()=>["X","Y","Z"][e.display.sdfAxis],Z),g(H,()=>$i(yr(e.display.sdfFraction)),null),g(ie,()=>$i(is),null),L})(),b(G,{get when(){return En(e.display.sdfView)},get children(){var L=Yh(),H=L.firstChild,Z=H.nextSibling,ie=Z.nextSibling,ce=ie.firstChild,Re=ce.nextSibling,ue=Re.nextSibling,ze=ie.nextSibling;return g(H,()=>e.display.sdfView==="gradient"?"|∇f| — 1.0 is an exact distance field":"f — signed distance, inside and out"),g(ce,()=>e.display.sdfView==="gradient"?"0.5":"inside"),g(Re,()=>e.display.sdfView==="gradient"?"1.0":"0"),g(ue,()=>e.display.sdfView==="gradient"?"1.5":"outside"),g(ze,()=>e.display.sdfView==="gradient"?"Contours every 0.1, heaviest at 1.0":"Contours at the grid spacing and a fifth of it, densest within two intervals of the surface"),D(Ue=>Oe(Z,"background",Fs())),L}})]}}),$),g(y,b(G,{get when(){return e.display.sdfView==="normal"},get children(){return Xh()}}),$),g(y,b(G,{get when(){return e.display.sdfView==="depth"},get children(){return Kh()}}),$),I.$$input=L=>e.onChange({isoOffset:Number(L.currentTarget.value)}),g(B,()=>$i(e.display.isoOffset),null),N.$$click=()=>e.onChange({isoOffset:0}),g(c,b(G,{get when(){return t()},get children(){var L=lf(),H=L.firstChild,Z=H.firstChild,ie=Z.nextSibling,ce=ie.nextSibling,Re=H.nextSibling;Re.firstChild;var ue=Re.nextSibling;ue.firstChild;var ze=ue.nextSibling,Ue=ze.firstChild,qe=Ue.nextSibling,Ne=ze.nextSibling,xe=Ne.firstChild,et=xe.nextSibling,Wt=et.firstChild,st=Wt.nextSibling,j=et.nextSibling,R=j.firstChild,U=R.firstChild,v=R.nextSibling;j.nextSibling;var k=Ne.nextSibling;k.firstChild;var C=k.nextSibling;C.firstChild;var M=C.nextSibling,W=M.firstChild,F=W.nextSibling;return g(Z,()=>o()?"Preset is active":"Current settings are unsaved"),ie.$$click=()=>e.onPresetSave(i().id),ce.$$click=()=>e.onPresetReset(i().id),g(Re,b(Xt,{options:pf,get value(){return e.display.flatShading},onSelect:O=>e.onChange({flatShading:O})}),null),g(ue,b(Xt,{options:ff,get value(){return e.display.shadows},onSelect:O=>e.onChange({shadows:O})}),null),g(ze,b(Xt,{options:mf,get value(){return e.quality},get onSelect(){return e.onQualityChange}}),qe),g(qe,r),st.$$input=O=>e.onChange({marchSteps:Number(O.currentTarget.value)}),P(st,"min",ts),P(st,"max",ns),g(R,a,U),v.$$click=()=>e.onChange({marchSteps:null}),g(v,s),g(Ne,b(ke,{get checked(){return e.display.refineHit},onChange:O=>e.onChange({refineHit:O}),testId:"toggle-refine-hit",get children(){return["Refine hit",Zh()]}}),null),g(k,b(ke,{get checked(){return e.display.cullBounds},onChange:O=>e.onChange({cullBounds:O}),testId:"toggle-cull-bounds",get children(){return["Bounds culling",Jh()]}}),null),g(C,b(ke,{get checked(){return e.pathTracing},get onChange(){return e.onPathTracingChange},testId:"toggle-path-tracing",get children(){return["Path tracing",Qh()]}}),null),g(C,b(ke,{get checked(){return e.display.xray>0},onChange:O=>e.onChange({xray:O?1:0}),testId:"toggle-xray",get children(){return["X-ray",ef()]}}),null),g(C,b(Ae,{each:bf,children:O=>b(ke,{get checked(){return!!e.display[O.key]},onChange:Y=>e.onChange({[O.key]:Y}),get testId(){return`toggle-${O.key}`},get children(){return[X(()=>O.label),b(G,{get when(){return O.tier},children:Y=>b(tr,{get kind(){return Y()}})}),(()=>{var Y=fa();return g(Y,()=>O.hint),Y})()]}})}),null),g(C,b(ke,{get checked(){return X(()=>e.display.showBrep===!0)()&&!Pi()},get disabled(){return Pi()},onChange:O=>e.onChange({showBrep:O}),testId:"toggle-showBrep",get children(){return["B-rep inspector",b(tr,{kind:"brep"}),b(G,{get when(){return Pi()},get fallback(){return uf()},get children(){return[b(ud,{kind:"brep"}),b(G,{get when(){return!ws("brep")},get children(){var O=tf();return g(O,An),O}})]}})]}}),null),g(M,b(ke,{get checked(){return e.display.showGraticule},onChange:O=>e.onChange({showGraticule:O}),testId:"toggle-graticule",get children(){return["Ground grid",nf()]}}),F),g(M,b(ke,{get checked(){return e.display.showOverlays},onChange:O=>e.onChange({showOverlays:O}),testId:"toggle-construction-overlay",get children(){return["Construction overlay",rf()]}}),F),g(M,b(ke,{get checked(){return e.display.showSketches},onChange:O=>e.onChange({showSketches:O}),testId:"toggle-showSketches",get children(){return["Sketch geometry",af()]}}),F),g(M,b(ke,{get checked(){return e.display.showAllPlanes===!0},onChange:O=>e.onChange({showAllPlanes:O}),testId:"toggle-showAllPlanes",get children(){return["All sketch planes",sf()]}}),F),g(M,b(ke,{get checked(){return e.display.showConstraints},onChange:O=>e.onChange({showConstraints:O}),testId:"toggle-constraints",get children(){return["Constraints",of()]}}),F),g(F,b(ke,{compact:!0,get checked(){return e.display.showFixedConstraints},get disabled(){return!e.display.showConstraints},onChange:O=>e.onChange({showFixedConstraints:O}),testId:"toggle-fixed-constraints",children:"Fixed badges"}),null),g(F,b(ke,{compact:!0,get checked(){return e.display.showDistanceConstraints},get disabled(){return!e.display.showConstraints},onChange:O=>e.onChange({showDistanceConstraints:O}),testId:"toggle-distance-constraints",children:"Distance dimensions"}),null),g(F,b(ke,{compact:!0,get checked(){return e.display.showConstraintValues},get disabled(){return!e.display.showConstraints||!e.display.showDistanceConstraints},onChange:O=>e.onChange({showConstraintValues:O}),testId:"toggle-constraint-values",children:"Dimension values"}),null),D(O=>{var Y=o(),z=e.display.marchSteps===null,te=!e.display.showConstraints;return Y!==O.e&&(ie.disabled=O.e=Y),z!==O.t&&(v.disabled=O.t=z),te!==O.a&&F.classList.toggle("disabled",O.a=te),O},{e:void 0,t:void 0,a:void 0}),D(()=>st.value=a()),L}}),null),D(L=>{var H=`render-customize ${t()?"active":""}`,Z=t(),ie=e.display.isoOffset===0;return H!==L.e&&Lt(p,L.e=H),Z!==L.t&&P(p,"aria-expanded",L.t=Z),ie!==L.a&&(N.disabled=L.a=ie),L},{e:void 0,t:void 0,a:void 0}),D(()=>I.value=e.display.isoOffset),l})()}Ce(["click","input"]);var wf=S("<div class=render-popover data-testid=render-popover>"),_f=S('<div class=render-popover-anchor><button type=button title="Render settings — presets, shading, overlays and quality"aria-label="Render settings"data-testid=display-options><span class=rail-word>Render');const[Jt,pa]=T(!1);function xf(e){let t;return ys({open:Jt,close:()=>pa(!1),within:()=>t,escape:!0}),(()=>{var n=_f(),i=n.firstChild,r=i.firstChild,a=t;return typeof a=="function"?on(a,n):t=n,i.$$click=()=>pa(!Jt()),g(i,b(Fc,{size:16}),r),g(n,b(G,{get when(){return Jt()},get children(){var s=wf();return g(s,b(yf,Le(()=>e.render))),s}}),null),D(s=>{var o=`viewport-bar-button ${Jt()?"active":""}`,l=Jt();return o!==s.e&&Lt(i,s.e=o),l!==s.t&&P(i,"aria-expanded",s.t=l),s},{e:void 0,t:void 0}),n})()}Ce(["click"]);const q=Math.fround,ee=(e,t,n)=>[q(e),q(t),q(n)],Ve=e=>{const t=e;return ee(t[0],t[1],t[2])},Ee=(e,t)=>ee(e[0]+t[0],e[1]+t[1],e[2]+t[2]),Je=(e,t)=>ee(e[0]-t[0],e[1]-t[1],e[2]-t[2]),lr=(e,t)=>ee(e[0]*t[0],e[1]*t[1],e[2]*t[2]),me=(e,t)=>ee(e[0]*t,e[1]*t,e[2]*t),nn=(e,t)=>ee(Math.min(e[0],t[0]),Math.min(e[1],t[1]),Math.min(e[2],t[2])),rn=(e,t)=>ee(Math.max(e[0],t[0]),Math.max(e[1],t[1]),Math.max(e[2],t[2])),Rn=e=>ee(Math.abs(e[0]),Math.abs(e[1]),Math.abs(e[2])),kf=e=>q(q(e[0]+e[1])+e[2]),Qe=(e,t)=>kf(lr(e,t)),ut=e=>q(Math.sqrt(e)),Se=e=>q(e),Ei=[0,0,0],Qn=e=>Je(e.center,e.half),ei=e=>Ee(e.center,e.half);function ma(e){if(!e.length||e.some(a=>a===null))return null;const t=e;let n=Qn(t[0]),i=ei(t[0]),r=t[0].scale;for(const a of t.slice(1))n=nn(n,Qn(a)),i=rn(i,ei(a)),r=Math.min(r,a.scale);return{center:me(Ee(n,i),.5),half:me(Je(i,n),.5),scale:r}}function Ii(e,t){if(e===null)return null;const n=e.scale===1?t:q(t/e.scale);return{center:e.center,half:Ee(e.half,[n,n,n]),scale:e.scale}}function ga(e,t,n){const i=ut(Math.max(Qe(n,n),q(1e-12))),r=ee(n[0]/i,n[1]/i,n[2]/i),a=ut(Qe(e.half,e.half)),s=Je(e.center,t),o=Qe(s,r),l=Je(s,me(r,o)),c=q(ut(Qe(l,l))+a),d=q(o-a),h=q(o+a),u=me(ee(...r.map(x=>ut(Math.max(q(1-q(x*x)),0)))),c),p=nn(me(r,d),me(r,h)),f=rn(me(r,d),me(r,h)),m=Je(Ee(t,p),u),w=Ee(Ee(t,f),u);return{center:me(Ee(m,w),.5),half:me(Je(w,m),.5),scale:e.scale}}const cr=(e,t)=>ee(Qe(e[0],t),Qe(e[1],t),Qe(e[2],t));function Sf(e,t){const n=Ve(e),i=ut(Math.max(Qe(n,n),q(1e-24)));return Vs(ee(n[0]/i,n[1]/i,n[2]/i),t)}function $f(e,t){const n=Ve(e),i=ut(Qe(n,n));return Vs(ee(n[0]/i,n[1]/i,n[2]/i),t)}function Vs(e,t){const n=q(t),i=q(Math.cos(n)),r=q(Math.sin(n)),a=q(1-i),[s,o,l]=e,c=(d,h)=>q(d*h);return[ee(q(c(c(a,s),s)+i),q(c(c(a,s),o)-c(r,l)),q(c(c(a,s),l)+c(r,o))),ee(q(c(c(a,s),o)+c(r,l)),q(c(c(a,o),o)+i),q(c(c(a,o),l)-c(r,s))),ee(q(c(c(a,s),l)-c(r,o)),q(c(c(a,o),l)+c(r,s)),q(c(c(a,l),l)+i))]}function Pf(e){const t=Ve(e),n=ut(Math.max(Qe(t,t),q(1e-24))),i=ee(t[0]/n,t[1]/n,t[2]/n);return[[1,0,0],[0,1,0],[0,0,1]].map((a,s)=>Je(a,me(i,q(2*i[s]))))}function va(e,t,n){return e===null?null:{center:Ee(cr(t,e.center),n),half:cr(t.map(Rn),e.half),scale:e.scale}}const gn=(e,t)=>(e.params[t]??[]).map(n=>ee(n[0],n[1],0)),ba=(e,t)=>t in e.params&&!(e.implicit??(e.params[t]===0?[t]:[])).includes(t);function Hs(e,t){const n=e.params,i=a=>({center:Ei,half:a,scale:1});switch(e.type){case"sphere":{const a=Math.abs(Se(n.radius));return i(ee(a,a,a))}case"box":return i(Rn(Ve(n.size)));case"round_box":{const a=Math.abs(Se(n.radius));return i(Ee(Rn(Ve(n.size)),[a,a,a]))}case"cylinder":{const a=Math.abs(Se(n.radius)),s=Math.abs(Se(n.height));return i(ee(a,a,s))}case"capsule":{const a=Math.abs(Se(n.radius)),s=Math.abs(Se(n.height));return i(ee(a,a,s+a))}case"torus":{const a=Math.abs(Se(n.major_radius)),s=Math.abs(Se(n.minor_radius));return i(ee(a+s,a+s,s))}case"extruded_polygon":{if(ba(e,"draft"))return null;const a=gn(e,"vertices"),s=q(Math.abs(Se(n.depth))*.5);if(ba(e,"twist")){const h=ut(a.map(u=>q(q(u[0]*u[0])+q(u[1]*u[1]))).reduce((u,p)=>Math.max(u,p)));return i(ee(h,h,s))}const o=a.reduce(nn),l=a.reduce(rn),c=me(Ee(o,l),.5),d=me(Je(l,o),.5);return{center:ee(c[0],c[1],0),half:ee(d[0],d[1],s),scale:1}}case"revolved_polygon":{const a=gn(e,"vertices"),s=Math.max(q(Math.max(...a.map(c=>c[0]))+Se(n.offset)),0),o=Math.min(...a.map(c=>c[1])),l=Math.max(...a.map(c=>c[1]));return{center:ee(0,q(q(o+l)*.5),0),half:ee(s,q(q(l-o)*.5),s),scale:1}}case"lofted_polygon":{const a=[...gn(e,"vertices"),...gn(e,"vertices_b")],s=a.reduce(nn),o=a.reduce(rn),l=me(Ee(s,o),.5),c=me(Je(o,s),.5);return{center:ee(l[0],l[1],0),half:ee(c[0],c[1],q(Math.abs(Se(n.height))*.5)),scale:1}}case"union":return Ii(ma(t),Math.max(q(Se(n.smoothness??.1)*4),q(1e-10)));case"difference":case"intersection":return t.length?t[0]:null;case"xor":return ma(t)}if(!t.length||t[0]===null)return null;const r=t[0];switch(e.type){case"translate":return{center:Ee(r.center,Ve(n.offset)),half:r.half,scale:r.scale};case"rotate":return va(r,Sf(n.axis,n.angle),Ei);case"scale":{const a=Ve(n.scale),s=Rn(a),o=Math.max(s[0],s[1],s[2]),l=Math.min(s[0],s[1],s[2]);return{center:lr(r.center,a),half:lr(r.half,s),scale:q(r.scale*q(l/o))}}case"mirror":{const a=Ve(n.origin),s=Pf(n.normal);return va(r,s,Je(a,cr(s,a)))}case"shell":return Ii(r,q(Math.abs(Se(n.thickness))*.5));case"offset":return Ii(r,Math.max(Se(n.distance),0));case"twist":return ga(r,Ei,Ve(n.axis));case"linear_pattern":{const a=Math.trunc(n.count);if(a<=1)return r;const s=Ve(n.direction),o=ut(Math.max(Qe(s,s),q(1e-24))),l=ee(s[0]/o,s[1]/o,s[2]/o),c=me(l,q(Se(n.spacing)*(a-1))),d=nn(Qn(r),Ee(Qn(r),c)),h=rn(ei(r),Ee(ei(r),c));return{center:me(Ee(d,h),.5),half:me(Je(h,d),.5),scale:r.scale}}case"polar_pattern":return Math.trunc(n.count)<=1?r:ga(r,Ve(n.origin),Ve(n.direction))}return null}function dr(e){try{return Hs(e,e.children.map(dr))}catch{return null}}const Ct={group:3,binding:0,addressSpace:"storage, read",slotBytes:16,reservedSlots:2,cullMargin:1e-4},Us={format:"cadjoint-scene",versions:[1,2]},vn={format:"cadjoint-demo",version:1},ya=Us.format,wa=Us.versions,Ef={color:[1,1,1],roughness:.5,metallic:0,opacity:1,ior:1,reflectivity:0};function If(e){if(e.format!==ya||!wa.includes(e.version))throw new Error(`scene is ${e.format} v${e.version}; this reads ${ya} v${wa.join(", v")}`);return e.root}function $e(e){return Af(e,9)}function Af(e,t){if(Number.isNaN(e))return"nan";if(e===1/0)return"inf";if(e===-1/0)return"-inf";if(e===0)return Object.is(e,-0)?"-0":"0";const n=e<0?"-":"",[i,r]=Mf(Math.abs(e),t);if(r<-4||r>=t){let s=i.length>1?`${i[0]}.${i.slice(1)}`:i;s=_a(s);const o=Math.abs(r);return`${n}${s}e${r<0?"-":"+"}${o<10?"0":""}${o}`}let a;if(r>=0){const s=i.slice(0,r+1).padEnd(r+1,"0"),o=i.slice(r+1);a=o?`${s}.${o}`:s}else a=`0.${"0".repeat(-r-1)}${i}`;return n+_a(a)}function Mf(e,t){const n=e.toExponential(99),[i,r]=n.split("e");let a=Number(r);const s=i.replace(".","");let o=s.slice(0,t);const l=s.slice(t);if(l[0]>"5"||l[0]==="5"&&(/[1-9]/.test(l.slice(1))||Number(o[o.length-1])%2===1)){const d=(BigInt(o)+1n).toString();d.length>t?(o=d.slice(0,t),a+=1):o=d}return[o,a]}function _a(e){return e.includes(".")?e.replace(/0+$/,"").replace(/\.$/,""):e}const{group:Cf,binding:Tf,addressSpace:Of,slotBytes:Ai,reservedSlots:Lf,cullMargin:Df}=Ct;class ur extends Error{}const Ar={sphere:["Sphere",{params:["radius"],arity:{radius:1},body:"    return length(p) - radius;"}],box:["Box",{params:["size"],arity:{size:3},body:`    let q = abs(p) - size;
    let max_q = max(q.x, max(q.y, q.z));
    let clamped = max(q, vec3<f32>(0.0));
    let squared = dot(clamped, clamped);
    if (max_q <= 0.0) { return max_q; }
    return sqrt(squared);`}],torus:["Torus",{params:["major_radius","minor_radius"],arity:{major_radius:1,minor_radius:1},body:`    let q_xy = length(p.xy) - major_radius;
    return length(vec2<f32>(q_xy, p.z)) - minor_radius;`}],cylinder:["Cylinder",{params:["radius","height"],arity:{radius:1,height:1},body:`    let d_xy = sqrt(p.x * p.x + p.y * p.y + 1e-20) - radius;
    let d_z = abs(p.z) - height;
    let max_d = max(d_xy, d_z);
    if (max_d <= 0.0) { return max_d; }
    let a = max(d_xy, 0.0);
    let b = max(d_z, 0.0);
    return sqrt(a * a + b * b);`}],capsule:["Capsule",{params:["radius","height"],arity:{radius:1,height:1},body:`    let z = clamp(p.z, -height, height);
    return length(vec3<f32>(p.x, p.y, p.z - z)) - radius;`}]},Rf="@group(1) @binding(0) var<storage, read> profile_vertices: array<vec2<f32>>;",Bf=`fn profile_vertex(i: u32) -> vec2<f32> {
    return sdf_parameters.values[i].xy;
}`,qs=`fn profile_vertex(i: u32) -> vec2<f32> {
    return profile_vertices[i];
}`,Ff=`fn profile_distance(p: vec2<f32>, offset: u32, count: u32) -> f32 {
    var d = dot(p - profile_vertex(offset), p - profile_vertex(offset));
    var s = 1.0;
    // The previous vertex is carried rather than re-read: one buffer load per
    // edge instead of two, and no modulo. Edge for edge the same arithmetic.
    var vj = profile_vertex(offset + count - 1u);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let vi = profile_vertex(offset + i);
        let e = vj - vi;
        let w = p - vi;
        let t = clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        let b = w - e * t;
        d = min(d, dot(b, b));
        let c1 = p.y >= vi.y;
        let c2 = p.y < vj.y;
        let c3 = (e.x * w.y) > (e.y * w.x);
        if ((c1 == c2) && (c2 == c3)) { s = -s; }
        vj = vi;
    }
    return s * sqrt(d);
}`,zf=`fn prim_extrudedpolygon(
    p: vec3<f32>, offset: u32, count: u32, depth: f32, slope: f32, twist: f32
) -> f32 {
    // \`slope\` is tan(draft) and \`twist\` is radians over the full depth, both
    // folded by the caller; zero is the plain extrusion and both terms
    // vanish, matching the Python's static skips.
    let theta = twist * p.z / depth;
    let c = cos(theta);
    let s = sin(theta);
    let xy = vec2<f32>(p.x * c + p.y * s, p.y * c - p.x * s);
    let d2 = profile_distance(xy, offset, count) + slope * (p.z + depth * 0.5);
    let dz = abs(p.z) - depth * 0.5;
    let max_d = max(d2, dz);
    if (max_d <= 0.0) { return max_d; }
    let a = max(d2, 0.0);
    let b = max(dz, 0.0);
    return sqrt(a * a + b * b);
}`,Nf=`fn on_axis(x: f32, axis: f32) -> bool {
    return abs(x - axis) <= 1e-6 * (1.0 + abs(x) + abs(axis));
}

fn profile_distance_revolved(p: vec2<f32>, offset: u32, count: u32, axis: f32) -> f32 {
    var d = 1e30;
    var s = 1.0;
    // The previous vertex is carried rather than re-read: one buffer load per
    // edge instead of two, and no modulo. Edge for edge the same arithmetic.
    var vj = profile_vertex(offset + count - 1u);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let vi = profile_vertex(offset + i);
        let e = vj - vi;
        let w = p - vi;
        if (!(on_axis(vi.x, axis) && on_axis(vj.x, axis))) {
            let t = clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
            let b = w - e * t;
            d = min(d, dot(b, b));
        }
        let c1 = p.y >= vi.y;
        let c2 = p.y < vj.y;
        let c3 = (e.x * w.y) > (e.y * w.x);
        if ((c1 == c2) && (c2 == c3)) { s = -s; }
        vj = vi;
    }
    return s * sqrt(d);
}`,Gf=`fn prim_revolvedpolygon(p: vec3<f32>, offset: u32, count: u32, radial: f32) -> f32 {
    let r = sqrt(p.x * p.x + p.z * p.z + 1e-20) - radial;
    return profile_distance_revolved(vec2<f32>(r, p.y), offset, count, -radial);
}`,jf=`fn rotate_about(p: vec3<f32>, origin: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let v = p - origin;
    let c = cos(angle);
    let s = sin(angle);
    let parallel = axis * dot(v, axis);
    return origin + v * c + cross(axis, v) * s + parallel * (1.0 - c);
}`,Vf=`fn profile_distance_lerped(
    p: vec2<f32>, a: u32, b: u32, count: u32, t: f32
) -> f32 {
    let first = mix(profile_vertex(a), profile_vertex(b), t);
    var d = dot(p - first, p - first);
    var s = 1.0;
    var vj = mix(profile_vertex(a + count - 1u), profile_vertex(b + count - 1u), t);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let vi = mix(profile_vertex(a + i), profile_vertex(b + i), t);
        let e = vj - vi;
        let w = p - vi;
        let u = clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        let q = w - e * u;
        d = min(d, dot(q, q));
        let c1 = p.y >= vi.y;
        let c2 = p.y < vj.y;
        let c3 = (e.x * w.y) > (e.y * w.x);
        if ((c1 == c2) && (c2 == c3)) { s = -s; }
        vj = vi;
    }
    return s * sqrt(d);
}`,Hf=`fn prim_loftedpolygon(
    p: vec3<f32>, a: u32, b: u32, count: u32, height: f32
) -> f32 {
    let t = clamp(p.z / height + 0.5, 0.0, 1.0);
    let d2 = profile_distance_lerped(p.xy, a, b, count, t);
    let dz = abs(p.z) - height * 0.5;
    let max_d = max(d2, dz);
    if (max_d <= 0.0) { return max_d; }
    let ea = max(d2, 0.0);
    let eb = max(dz, 0.0);
    return sqrt(ea * ea + eb * eb);
}`,Uf=`fn material_base(p: vec3<f32>) -> vec4<f32> {
    let m = material_block_impl(p);
    return vec4<f32>(m[0].x, m[1].x, m[2].x, m[3].x);
}

fn material_optics(p: vec3<f32>) -> vec4<f32> {
    let m = material_block_impl(p);
    return vec4<f32>(m[0].y, m[1].y, m[2].y, m[3].y);
}`,qf=`fn cull_box(p: vec3<f32>, center: vec3<f32>, half_size: vec3<f32>) -> f32 {
    let q = max(abs(p - center) - half_size, vec3<f32>(0.0));
    return sqrt(dot(q, q));
}`,Ws="__CADJOINT_CULL_MARGIN__",Ys={profile_distance:Ff,profile_distance_revolved:Nf,profile_vertex:qs,profile_distance_lerped:Vf,prim_loftedpolygon:Hf,prim_extrudedpolygon:zf,prim_revolvedpolygon:Gf,rotate_about:jf,cull_box:qf};for(const[e,t]of Object.values(Ar)){const n=e.toLowerCase(),i=t.params.map(r=>`${r}: ${t.arity[r]===1?"f32":"vec3<f32>"}`).join(", ");Ys[`prim_${n}`]=`fn prim_${n}(p: vec3<f32>, ${i}) -> f32 {
${t.body}
}`}const bn={translate:[["offset"],{offset:3},(e,t)=>`${e} - ${t.offset}`]},hr=new Set(["union","intersection","difference"]),Xs=new Set(["extruded_polygon","revolved_polygon","lofted_polygon"]),Wf=[...Object.keys(Ar),...Xs,"translate","rotate","mirror","shell","offset","linear_pattern","polar_pattern",...hr];function fe(e,t){const n=e.params[t];if(n===void 0)throw new ur(`${e.type} has no parameter ${t}`);return(Array.isArray(n)?n.flat():[n]).map(i=>i===null?NaN:Number(i))}function Xe(e,t){return t===1?$e(e[0]):`vec${t}<f32>(${e.slice(0,t).map($e).join(", ")})`}function xa(e,t){const n=e.free?.[t];return typeof n=="string"?n:null}function zt(e,t){return t in e.params?e.type!=="extruded_polygon"||t!=="draft"&&t!=="twist"?!0:e.implicit?!e.implicit.includes(t):e.params[t]!==0:!1}const Ke=e=>e.children;function ka(e){return Math.sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2])}class ti{derived=[];depends={};slots=[];functions=[];vertices=[];bounds=new Map;boxes=new Map;reads=new Map;slotOf=new Map;kernels=new Set;nodes=new Map;materials=new Map;sketches=new Map;anonymous=new WeakMap;anonymousCount=0;uniforms;culling;constructor(t,n){this.uniforms=t,this.culling=n}key(t){if(t.id!=null)return t.id;let n=this.anonymous.get(t);return n===void 0&&(n=`#${this.anonymousCount++}`,this.anonymous.set(t,n)),n}vertexSlice(t,n){const i=n==="v"?"vertices":"vertices_b",r=t.params[i]??[],a=t.free?.[i]??[];if(this.uniforms){let o=null;return r.forEach((l,c)=>{const d=a[c]??null,h=this.key(t);this.sketches.has(h)||this.sketches.set(h,this.sketches.size);const u=this.sketches.get(h),p=d??`sketch${u}.${n}${c}`;this.slotOf.has(p)||(this.slotOf.set(p,this.slots.length),this.slots.push([p,2,[Number(l[0]),Number(l[1])]]));const f=this.slotOf.get(p);o===null&&(o=f)}),[o??0,r.length]}const s=this.vertices.length;for(const o of r)this.vertices.push([Number(o[0]),Number(o[1])]);return[s,r.length]}read(t,n,i){const r=fe(t,n);if(!this.uniforms)return Xe(r,i);const a=xa(t,n);return a===null?Xe(r,i):(this.slotOf.has(a)||(this.slotOf.set(a,this.slots.length),this.slots.push([a,i,r.slice(0,i)])),`sdf_parameters.values[${this.slotOf.get(a)}].${"xyzw".slice(0,i)}`)}need(...t){for(const n of t)this.kernels.has(n)||(this.kernels.add(n),this.functions.push(Ys[n]))}functionFor(t){const n=this.key(t),i=this.nodes.get(n);if(i!==void 0)return i;const r=`node${this.nodes.size}`;this.nodes.set(n,r);const a=this.body(t);return this.functions.push(`fn ${r}(p: vec3<f32>) -> f32 {
${a}
}`),r}materialFor(t){const n=this.key(t),i=this.materials.get(n);if(i!==void 0)return i;const r=`mat${this.materials.size}`;this.materials.set(n,r);const a=this.materialBody(t);return this.functions.push(`fn ${r}(p: vec3<f32>) -> mat4x2<f32> {
${a}
}`),r}materialBody(t){const n=t.type;if(hr.has(n)){const c=Ke(t),d=zt(t,"smoothness")?fe(t,"smoothness")[0]:0,h=$e(Math.max(d*4,1e-10)),u=[`    var acc = ${this.functionFor(c[0])}(p);`,`    var m = ${this.materialFor(c[0])}(p);`];return c.slice(1).forEach((p,f)=>{u.push(`    let d${f} = ${this.functionFor(p)}(p);`,`    let n${f} = ${this.materialFor(p)}(p);`,`    let t${f} = clamp(0.5 + 0.5 * (d${f} - acc) / ${h}, 0.0, 1.0);`,`    m = m * t${f} + n${f} * (1.0 - t${f});`,`    let h${f} = max(${h} - abs(acc - d${f}), 0.0);`,`    acc = min(acc, d${f}) - h${f} * h${f} * 0.25 / ${h};`)}),u.push("    return m;"),u.join(`
`)}if(n in bn||["rotate","mirror","shell","offset"].includes(n))return`    return ${this.materialFor(Ke(t)[0])}(${this.movedPoint(t)});`;if(n==="linear_pattern"||n==="polar_pattern")return`    return ${this.materialFor(Ke(t)[0])}(p);`;const i=t.material??Ef,r=c=>c===null?NaN:c,a=i.color.map(r),s=[a[0],a[1],a[2],r(i.roughness)],o=[i.metallic,i.opacity,i.ior,i.reflectivity].map(r);return`    return mat4x2<f32>(${[0,1,2,3].map(c=>`vec2<f32>(${$e(s[c])}, ${$e(o[c])})`).join(", ")});`}rotationLiteral(t){return $f(fe(t,"axis"),fe(t,"angle")[0]).map(i=>`vec3<f32>(${i.map($e).join(", ")})`).join(", ")}movedPoint(t){const n=t.type;if(n==="shell"||n==="offset")return"p";if(n==="rotate")return`mat3x3<f32>(${this.rotationLiteral(t)}) * p`;if(n==="mirror"){const o=Xe(fe(t,"origin"),3),l=Xe(fe(t,"normal"),3);return`p - 2.0 * dot(p - ${o}, ${l}) * ${l}`}const[i,r,a]=bn[n],s=Object.fromEntries(i.map(o=>[o,this.read(t,o,r[o])]));return a("p",s)}nodeBoundsOf(t){const n=this.key(t);if(this.bounds.has(n))return this.bounds.get(n);this.bounds.set(n,null);const i=Ke(t).map(a=>this.nodeBoundsOf(a));let r;try{r=Hs(t,i)}catch{r=null}return r!==null&&([...r.center,...r.half].every(Number.isFinite)&&r.scale>0||(r=null)),this.bounds.set(n,r),r}boxFor(t){if(!this.culling)return null;const n=this.key(t),i=this.boxes.get(n);if(i!==void 0)return i;const r=this.nodeBoundsOf(t);if(r===null)return null;this.need("cull_box");let a;if(this.uniforms){const s=this.boxes.size,o=[...this.slotsRead(t)].sort(),l=[];for(const[c,d]of[["center",r.center],["half",r.half]]){const h=`__cull${s}.${c}`;this.slotOf.set(h,this.slots.length),this.slots.push([h,3,[...d]]),this.derived.push(h),this.depends[h]=o,l.push(`sdf_parameters.values[${this.slotOf.get(h)}].xyz`)}a=[l[0],l[1],r.scale]}else a=[Xe(r.center,3),Xe(r.half,3),r.scale];return this.boxes.set(n,a),a}slotsRead(t){const n=this.key(t),i=this.reads.get(n);if(i!==void 0)return i;const r=new Set,a=this.sketches.get(n),s=(o,l)=>{o!==null&&this.slotOf.has(o)?r.add(o):a!==void 0&&this.slotOf.has(`sketch${a}.${l}`)&&r.add(`sketch${a}.${l}`)};for(const[o,l]of[["vertices","v"],["vertices_b","w"]]){const c=t.params[o]??[],d=t.free?.[o]??[];c.forEach((h,u)=>s(d[u]??null,`${l}${u}`))}for(const o of Object.keys(t.params))o==="vertices"||o==="vertices_b"||!zt(t,o)||s(xa(t,o),o);for(const o of Ke(t))for(const l of this.slotsRead(o))r.add(l);return this.reads.set(n,r),r}static boxTest(t,n,i){const[r,a,s]=n;let o=`cull_box(${t}, ${r}, ${a})`;return s!==1&&(o=`${$e(s)} * ${o}`),`${o} >= ${i} + ${Ws}`}body(t){const n=t.type,i=Ar[n];if(i){const[r,a]=i,s=r.toLowerCase();this.need(`prim_${s}`);const o=a.params.map(l=>this.read(t,l,a.arity[l])).join(", ");return`    return prim_${s}(p, ${o});`}if(Xs.has(n))return this.profileBody(t);if(n==="shell"||n==="offset"){const r=this.functionFor(Ke(t)[0]);return n==="shell"?`    return abs(${r}(p)) - ${$e(fe(t,"thickness")[0]/2)};`:`    return ${r}(p) - ${$e(fe(t,"distance")[0])};`}if(n==="mirror"){const r=this.functionFor(Ke(t)[0]),a=Xe(fe(t,"origin"),3);return`    let n = ${Xe(fe(t,"normal"),3)};
    return ${r}(p - 2.0 * dot(p - ${a}, n) * n);`}if(n==="rotate"){const r=this.rotationLiteral(t);return`    return ${this.functionFor(Ke(t)[0])}(mat3x3<f32>(${r}) * p);`}if(n in bn){const[r,a,s]=bn[n],o=Object.fromEntries(r.map(c=>[c,this.read(t,c,a[c])]));return`    return ${this.functionFor(Ke(t)[0])}(${s("p",o)});`}if(n==="linear_pattern"||n==="polar_pattern")return this.patternBody(t);if(hr.has(n))return this.booleanBody(t);throw new ur(`${n} has no direct WGSL kernel. Supported: ${Wf.join(", ")}`)}culledBooleanBody(t,n,i,r){const a=[`    var d = ${i[0]}(p);`];for(let s=1;s<n.length;s++){const o=i[s];let l,c,d;t==="union"?(l=`${o}(p)`,c=`max(d + ${r}, 0.0)`,d=`d = min(d, e${s}) - h${s} * h${s} * 0.25 / ${r};`):(l=`(-${o}(p))`,c=`max(${r} - d, 0.0)`,d=`d = max(d, e${s}) + h${s} * h${s} * 0.25 / ${r};`);const h=[`let e${s} = ${l};`,`let h${s} = max(${r} - abs(d - e${s}), 0.0);`,d],u=this.boxFor(n[s]);u===null?a.push(...h.map(p=>`    ${p}`)):(a.push(`    if (!(${ti.boxTest("p",u,c)})) {`),a.push(...h.map(p=>`        ${p}`)),a.push("    }"))}return a.push("    return d;"),a.join(`
`)}profileBody(t){if(t.type==="extruded_polygon"){this.need("profile_vertex","profile_distance","prim_extrudedpolygon");const[a,s]=this.vertexSlice(t,"v"),o=this.read(t,"depth",1);let l=0;if(zt(t,"draft")){const d=fe(t,"draft")[0]*(Math.PI/180);l=Math.sin(d)/Math.cos(d)}let c=0;return zt(t,"twist")&&(c=fe(t,"twist")[0]*(Math.PI/180)),`    return prim_extrudedpolygon(p, ${a}u, ${s}u, ${o}, ${$e(l)}, ${$e(c)});`}if(t.type==="lofted_polygon"){this.need("profile_vertex","profile_distance_lerped","prim_loftedpolygon");const[a,s]=this.vertexSlice(t,"v"),[o,l]=this.vertexSlice(t,"w");if(s!==l)throw new ur("A loft needs both profiles to have the same vertex count.");const c=this.read(t,"height",1);return`    return prim_loftedpolygon(p, ${a}u, ${o}u, ${s}u, ${c});`}this.need("profile_vertex","profile_distance_revolved","prim_revolvedpolygon");const[n,i]=this.vertexSlice(t,"v"),r=this.read(t,"offset",1);return`    return prim_revolvedpolygon(p, ${n}u, ${i}u, ${r});`}patternBody(t){const n=Ke(t)[0],i=this.functionFor(n),r=Math.trunc(fe(t,"count")[0]),a=zt(t,"skip_mask")?Math.trunc(fe(t,"skip_mask")[0]):0,s=[];for(let c=0;c<r;c++)Number(BigInt(a)>>BigInt(c)&1n)||s.push(c);if(s.length<=1)return`    return ${i}(p);`;const o=[`    var d = ${i}(p);`];if(o.push(`    let instances = array<i32, ${s.length-1}>(${s.slice(1).join(", ")});`),o.push(`    for (var k = 0; k < ${s.length-1}; k = k + 1) {`),o.push("        let i = f32(instances[k]);"),t.type==="linear_pattern"){const c=fe(t,"direction"),d=ka(c)||1,h=c.map(p=>p/d),u=fe(t,"spacing")[0];o.push(`        let q = p - ${Xe(h,3)} * (${$e(u)} * i);`)}else{this.need("rotate_about");const c=Xe(fe(t,"origin"),3),d=fe(t,"direction"),h=ka(d)||1,u=d.map(p=>p/h);o.push(`        let theta = -6.283185307179586 * i / ${$e(r)};`),o.push(`        let q = rotate_about(p, ${c}, ${Xe(u,3)}, theta);`)}const l=this.boxFor(n);return l===null?o.push(`        d = min(d, ${i}(q));`):(o.push(`        if (!(${ti.boxTest("q",l,"max(d, 0.0)")})) {`),o.push(`            d = min(d, ${i}(q));`),o.push("        }")),o.push("    }"),o.push("    return d;"),o.join(`
`)}booleanBody(t){const n=t.type,i=zt(t,"smoothness")?fe(t,"smoothness")[0]:0,r=$e(Math.max(i*4,1e-10)),a=Ke(t),s=a.map(l=>this.functionFor(l));if((n==="union"||n==="difference")&&this.culling)return this.culledBooleanBody(n,a,s,r);const o=[`    var d = ${s[0]}(p);`];for(const l of s.slice(1)){const c=n==="difference"?`(-${l}(p))`:`${l}(p)`,d=n==="union"?"min":"max",h=n==="union"?"-":"+";o.push(`    let e = ${c};`),o.push(`    let h = max(${r} - abs(d - e), 0.0);`),o.push(`    d = ${d}(d, e) ${h} h * h * 0.25 / ${r};`)}return o.push("    return d;"),Yf(o).join(`
`)}}function Yf(e){const t=[];let n=0;for(const i of e)i.trim().startsWith("let e =")&&(n+=1),t.push(i.replaceAll(" e ",` e${n} `).replaceAll(" h ",` h${n} `).replaceAll("(d - e)",`(d - e${n})`).replaceAll("d, e)",`d, e${n})`));return t}function Xf(e){const t=i=>String(i).padStart(6," "),n=[`// ${e.length} free parameters, one vec4<f32> slot each.`];return e.forEach(([i,r],a)=>{n.push(`// ${t(a*Ai)}  ${r}f  free   ${i}`)}),n.push(`// ${t(e.length*Ai)}  1f  reserved  NaN (unread)`),n.push(`// ${t((e.length+1)*Ai)}  1f  reserved  cull margin`),n.push("struct SdfParameters {"),n.push(`    values: array<vec4<f32>, ${e.length+Lf}>,`),n.push("};"),n.push(`@group(${Cf}) @binding(${Tf}) var<${Of}> sdf_parameters: SdfParameters;`),n.join(`
`)}function Kf(e,t={}){const{entryPoint:n="sdf",uniforms:i=!1,culling:r=!1}=t,a=If(e),s=new ti(i,r),o=s.functionFor(a),l=s.materialFor(a),c=[`fn ${n}(p: vec3<f32>) -> f32 {
    return ${o}(p);
}`,`fn material_block_impl(p: vec3<f32>) -> mat4x2<f32> {
    return ${l}(p);
}`,Uf].join(`

`);let d=[...s.functions];i?(d=d.map(u=>u===qs?Bf:u),d.unshift(Xf(s.slots))):s.vertices.length&&d.unshift(Rf);const h=i?`sdf_parameters.values[${s.slots.length+1}].x`:$e(Df);return{wgsl:[...d,c].join(`

`).replaceAll(Ws,h),vertices:s.vertices,parameters:s.slots,derived:s.derived,depends:s.depends}}const Mi="__CADJOINT_SDF_CODE__",Ci="__CADJOINT_SCENE_CODE__",Zf=`
fn material_base(_p: vec3<f32>) -> vec4<f32> {
  return vec4<f32>(0.85, 0.85, 0.85, 0.45);
}

fn material_optics(_p: vec3<f32>) -> vec4<f32> {
  return vec4<f32>(0.0, 1.0, 1.5, 0.0);
}
`,Jf=`__CADJOINT_SDF_CODE__

const PI: f32 = 3.141592653589793;

// display.x  projection: 0 perspective, 1 orthographic
// display.y  orthographic viewport height in world units
// display.z  DISPLAY_* flag bits, packed as a float
// display.w  x-ray strength, 0 disables
//
// The SDF views ride in the six scalars this struct already carried and never
// used, rather than in an eighth vec4. The struct is a contract with two
// writers — the playground renderer and the notebook widget in \`widget.py\` —
// and the widget allocates exactly 112 bytes, so growing it would fault the
// widget rather than merely ignore the new fields. Every one of these six is
// written as 0 by the widget, and 0 is the inert value of each:
//
// resolution.z     SDF view: 0 solid, 1 signed slice, 2 gradient magnitude
// resolution.w     slice plane axis: 0 X, 1 Y, 2 Z
// camera_pos.w     slice plane coordinate, world units
// camera_target.w  the level set traced: f = c rather than f = 0
// bg_color.w       isoline spacing on the slice, world units; 0 draws none
// path_settings.w  sphere-tracing step budget; 0 means the default
struct Uniforms {
  resolution   : vec4<f32>,
  camera_pos   : vec4<f32>,
  camera_target: vec4<f32>,
  light_dir    : vec4<f32>,
  bg_color     : vec4<f32>,
  path_settings: vec4<f32>,
  display      : vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

const SDF_VIEW_SOLID: f32 = 0.0;
const SDF_VIEW_SLICE: f32 = 1.0;
const SDF_VIEW_GRADIENT: f32 = 2.0;
const SDF_VIEW_NORMAL: f32 = 3.0;
const SDF_VIEW_DEPTH: f32 = 4.0;

/// True for the two views that replace the solid's shading rather than cutting
/// a plane through the field.
fn sdf_view_shades_surface(mode: f32) -> bool {
  return mode == SDF_VIEW_NORMAL || mode == SDF_VIEW_DEPTH;
}

/// The default march, kept identical to what this shader used before the
/// budget became a uniform, so a caller that writes 0 gets the old picture.
const DEFAULT_TRACE_STEPS: i32 = 96;

fn trace_steps() -> i32 {
  return select(i32(u.path_settings.w), DEFAULT_TRACE_STEPS, u.path_settings.w < 1.0);
}

/**
 * The field the whole shader traces, shades and shadows against.
 *
 * Every read of the scene goes through here rather than through \`sdf\`
 * directly, so the isosurface offset is not a shell drawn over the real
 * surface: the primary ray, the normal and both shadow marches all agree on
 * where the surface is, and \`f = c\` lights and occludes like the solid it is.
 * Subtracting a constant leaves the field's gradient untouched, so the march
 * stays as conservative as it was.
 */
fn scene_field(p: vec3<f32>) -> f32 {
  return sdf(p) - u.camera_target.w;
}

const DISPLAY_SHADOWS: u32 = 1u;
const DISPLAY_REFLECTIONS: u32 = 2u;
const DISPLAY_FLAT: u32 = 4u;
const DISPLAY_HIDE_SOLID: u32 = 8u;
const DISPLAY_HARD_SHADOWS: u32 = 16u;
// Secant refinement of the accepted hit; see \`refine_hit\`. A new *bit* rather
// than a new field: \`display.z\` is a bitfield in a struct that is already the
// full 112 bytes the notebook widget allocates, and an f32 carries integers
// exactly to 2^24, so bits are the one thing here that is still free.
const DISPLAY_REFINE_HIT: u32 = 32u;
/// Cap the solid where the section plane cuts it — see \`traced_field\`.
const DISPLAY_SECTION: u32 = 64u;

/// How much a section cap is darkened against the surface it cuts through.
const SECTION_CAP_TONE: f32 = 0.72;

/// How much of the sky the ground half of the hemisphere returns. 0.40, down
/// from 0.58, so a face pointing down reads darker than one pointing up.
const GROUND_BOUNCE: f32 = 0.40;
/// The key's gain in full shading, over the shared intensity the path tracer
/// also reads: the preview's single shadowed lamp has to carry the form that
/// the converged image gets from bounce light.
const KEY_GAIN: f32 = 2.4;
// Ambient occlusion: how hard a crowded sample darkens, and how dark a
// cavity may go — a bore is darker than its rim, never a hole.
const AO_STRENGTH: f32 = 6.0;
const AO_FLOOR: f32 = 0.45;

/// Half-width of the band the march accepts as "on the surface".
const SURFACE_EPS: f32 = 0.001;

/// Where a primary ray gives up and reports a miss, in world units.
const MAX_TRACE_DISTANCE: f32 = 100.0;

/// Secant iterations \`refine_hit\` may take. Each is one field evaluation, and
/// only pixels that already hit pay for any of them.
const REFINE_STEPS: i32 = 4;

// Hard shadows stay lifted rather than black: a single occlusion test has no
// penumbra to soften the edge, and fully dark cores hide the geometry inside.
const HARD_SHADOW_FLOOR: f32 = 0.45;

fn display_flag(flag: u32) -> bool {
  return (u32(max(u.display.z, 0.0)) & flag) != 0u;
}

/**
 * The section plane as a half-space, signed positive on the side removed.
 *
 * The plane is the one the slice controls already position — same axis
 * (\`resolution.w\`) and same coordinate (\`camera_pos.w\`) — so a reader who has
 * placed a slice does not place it twice.
 *
 * **Which half is removed depends on the camera, and that is deliberate.**
 * The design language forbids a *measurable* ground that moves with the eye:
 * you cannot read a value off a scale that changes as you orbit. A cutaway is
 * not a scale. Its whole purpose is to open the solid toward the reader, and
 * a section that keeps the near half shows you the outside of the far shell —
 * which is the "hollow" complaint this mode exists to answer. So the half the
 * camera is on is the half that goes, and orbiting through the plane swaps
 * them, exactly as it does in every drafting package.
 */
fn section_halfspace(p: vec3<f32>) -> f32 {
  let axis = i32(round(clamp(u.resolution.w, 0.0, 2.0)));
  let coordinate = u.camera_pos.w;
  // +1 when the camera sits on the positive side, so that side is removed.
  let side = select(-1.0, 1.0, u.camera_pos[axis] > coordinate);
  return (p[axis] - coordinate) * side;
}

/**
 * The field the tracer marches: the scene, capped at the section plane.
 *
 * \`max(a, b)\` is the intersection of two fields, and intersecting the solid
 * with a half-space is exactly what a section is. Three things fall out of
 * that one operation rather than having to be built:
 *
 * - **The cap is a real surface.** Clipping alone — refusing to draw in front
 *   of the plane — leaves the ray to carry on and hit the *inside* of the far
 *   shell, which is what makes a clipped solid look hollow. Intersecting
 *   instead means the ray stops *on* the plane, because that is where the
 *   field reaches zero.
 * - **The cap has the right normal.** \`sdf_normal\` takes central differences
 *   of this function, and on the cap the half-space is what the \`max\`
 *   selected, so the differences return the plane's own normal with no
 *   special case anywhere.
 * - **The cap has the right material.** \`material_base\` is still evaluated at
 *   the hit position, which is inside the solid that was cut, so the cut face
 *   arrives in the colour of the thing it is a cut through.
 *
 * The march stays correct: both operands are proper distance fields, and the
 * \`max\` of two of those never overestimates the distance to their
 * intersection, which is the only property sphere tracing needs.
 *
 * \`scene_field\` itself is left alone, because the data views read it. A
 * gradient view of a sectioned field would show the plane's own gradient and
 * report the scene as metric where it is not.
 */
fn traced_field(p: vec3<f32>) -> f32 {
  let field = scene_field(p);
  if (!display_flag(DISPLAY_SECTION)) { return field; }
  return max(field, section_halfspace(p));
}

/// True at a hit the section plane made, rather than the solid's own surface.
fn on_section_cap(p: vec3<f32>) -> bool {
  if (!display_flag(DISPLAY_SECTION)) { return false; }
  // The half-space won the \`max\`, within the band the march accepts.
  return section_halfspace(p) >= scene_field(p) - SURFACE_EPS;
}

// Only referenced by fs_main_depth, so pipelines built from vs_main/fs_main
// (the notebook widget) never see this binding in their derived layout.
struct ViewUniforms {
  view_proj : mat4x4<f32>,
};
@group(0) @binding(2) var<uniform> view: ViewUniforms;


fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
  return v * inverseSqrt(max(dot(v, v), 1e-12));
}

struct CameraBasis {
  forward : vec3<f32>,
  right   : vec3<f32>,
  up      : vec3<f32>,
};

// Orthonormal camera frame. World up is +Z — the library is Z-up, since a
// SketchPlane's default normal is +Z and sketches lie on the XY floor — except
// when the view direction is almost parallel to it: looking straight down is
// exactly what the Top preset asks for, and cross(forward, +Z) is degenerate
// there.
fn camera_basis(camera: vec3<f32>, look_at: vec3<f32>) -> CameraBasis {
  var basis: CameraBasis;
  basis.forward = safe_normalize(look_at - camera);
  var reference = vec3<f32>(0.0, 0.0, 1.0);
  if (abs(basis.forward.z) > 0.999) {
    reference = vec3<f32>(0.0, 1.0, 0.0);
  }
  basis.right = safe_normalize(cross(basis.forward, reference));
  basis.up = cross(basis.right, basis.forward);
  return basis;
}

struct PrimaryRay {
  origin    : vec3<f32>,
  direction : vec3<f32>,
};

// Ray through a pixel, for either projection.
//
// \`uv\` is the screen coordinate in the viewer's convention:
//   uv = (frag.xy / resolution - 0.5) * vec2(aspect, -1)
//
// Perspective rays fan out from the camera with a half-height of
// FOV_SCALE / 2 per unit of depth; orthographic rays are parallel and offset
// across a viewport \`ortho_height\` world units tall, so a scene keeps its
// framing when \`ortho_height\` is the perspective frustum height at the orbit
// distance.
fn primary_ray(
  uv: vec2<f32>,
  camera: vec3<f32>,
  look_at: vec3<f32>,
  projection: f32,
  ortho_height: f32,
) -> PrimaryRay {
  let basis = camera_basis(camera, look_at);
  var ray: PrimaryRay;
  if (projection > 0.5) {
    ray.origin = camera + (uv.x * basis.right + uv.y * basis.up) * ortho_height;
    ray.direction = basis.forward;
  } else {
    ray.origin = camera;
    ray.direction = safe_normalize(
      basis.forward + 1.5 * (uv.x * basis.right + uv.y * basis.up),
    );
  }
  return ray;
}


fn sdf_normal(p: vec3<f32>) -> vec3<f32> {
  let e = 0.001;
  // \`traced_field\`, not \`scene_field\`: on a section cap the half-space is
  // what the \`max\` selected, so these differences return the plane's own
  // normal and the cut face lights like the flat face it is.
  return safe_normalize(vec3<f32>(
    traced_field(p + vec3<f32>( e, 0.0, 0.0)) - traced_field(p + vec3<f32>(-e, 0.0, 0.0)),
    traced_field(p + vec3<f32>(0.0,  e, 0.0)) - traced_field(p + vec3<f32>(0.0, -e, 0.0)),
    traced_field(p + vec3<f32>(0.0, 0.0,  e)) - traced_field(p + vec3<f32>(0.0, 0.0, -e)),
  ));
}

// |∇f| by the same central differences, at a step scaled to what is on screen.
//
// This is the number that says whether the field is still a *distance*. An
// exact SDF has |∇f| = 1 everywhere it is differentiable; a smooth union, an
// offset or a scaled primitive breaks that, and where it is broken the
// mesher's Hermite data and the seam projections are being fed a lie. Drawn
// as a deviation from 1, the non-metric regions are exactly the coloured ones.
fn sdf_gradient_magnitude(p: vec3<f32>, e: f32) -> f32 {
  return length(vec3<f32>(
    scene_field(p + vec3<f32>( e, 0.0, 0.0)) - scene_field(p + vec3<f32>(-e, 0.0, 0.0)),
    scene_field(p + vec3<f32>(0.0,  e, 0.0)) - scene_field(p + vec3<f32>(0.0, -e, 0.0)),
    scene_field(p + vec3<f32>(0.0, 0.0,  e)) - scene_field(p + vec3<f32>(0.0, 0.0, -e)),
  )) / (2.0 * e);
}

/**
 * Tighten an accepted hit onto the surface, from the two samples the march
 * already holds.
 *
 * Sphere tracing stops at the first sample whose |f| is inside
 * \`SURFACE_EPS\` and takes that sample's \`t\` as the hit. On a ray meeting the
 * surface squarely that is accurate to the band. On a grazing one it is not:
 * the step is \`0.9|f|\`, and along a near-tangent |f| falls slowly, so the
 * march creeps and stops while still up to a whole epsilon short. The error
 * lies along the ray, which is why it shows as a bevelled, crawling
 * silhouette and as normals sampled off the surface rather than on it.
 *
 * \`f\` restricted to the ray is a scalar function of \`t\`, so this is just root
 * finding, and the march has already produced two samples of it. The secant
 * rule takes the first correction for free — no field evaluation at all —
 * and each further step costs one.
 *
 * The clamp is the whole safety argument. \`f\` along a ray is only locally
 * linear, and an unclamped secant step on a non-monotone field (a thin wall,
 * a blend fillet) can land anywhere, including behind the camera. Confining
 * every step to the bracket the march established, widened by one epsilon,
 * means refinement can move the hit by at most an epsilon in either
 * direction — which is exactly the error it exists to remove, and never more.
 * A hit is never created, destroyed or moved to another surface: silhouettes
 * are identical with this on and off.
 *
 * The path tracer has done the same thing since it was written
 * (\`refine_sign_crossing\`, seven bisections on a sign bracket). This is the
 * preview catching up, which is also why the two used to disagree about where
 * a grazing surface was.
 */
fn refine_hit(
  ro: vec3<f32>,
  rd: vec3<f32>,
  t_prev: f32,
  d_prev: f32,
  t_hit: f32,
  d_hit: f32,
) -> f32 {
  var ta = t_prev;
  var da = d_prev;
  var tb = t_hit;
  var db = d_hit;
  let low = min(t_prev, t_hit) - SURFACE_EPS;
  let high = max(t_prev, t_hit) + SURFACE_EPS;
  for (var i = 0; i < REFINE_STEPS; i++) {
    let denom = db - da;
    // Two samples of equal value say nothing about where the root is.
    if (abs(denom) < 1e-9) { break; }
    let tc = clamp(tb - db * (tb - ta) / denom, low, high);
    let dc = traced_field(ro + rd * tc);
    ta = tb;
    da = db;
    tb = tc;
    db = dc;
    if (abs(dc) < SURFACE_EPS * 0.01) { break; }
  }
  return tb;
}

fn trace(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
  var t = 0.01;
  // The sample before the current one, which is the second point the secant
  // rule needs. Held whether or not refinement is on: the branch is on a
  // uniform, so it is coherent across the whole draw, and the bookkeeping is
  // two registers rather than a divergent path.
  var t_prev = t;
  var d_prev = 0.0;
  var have_prev = false;
  let refine = display_flag(DISPLAY_REFINE_HIT);
  let budget = trace_steps();
  for (var i = 0; i < budget; i++) {
    let d = traced_field(ro + rd * t);
    if (abs(d) < SURFACE_EPS) {
      if (refine && have_prev) {
        return refine_hit(ro, rd, t_prev, d_prev, t, d);
      }
      return t;
    }
    if (t > MAX_TRACE_DISTANCE) { return -1.0; }
    t_prev = t;
    d_prev = d;
    have_prev = true;
    t += max(abs(d) * 0.9, 0.0005);
  }
  return -1.0;
}

// One occlusion ray, no penumbra: cheaper than soft_shadow and gives the crisp
// edge a drafting view wants.
fn hard_shadow(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
  var t = 0.02;
  let budget = trace_steps() / 2;
  for (var i = 0; i < budget; i++) {
    let h = traced_field(ro + rd * t);
    if (h < 0.001) { return 0.0; }
    t += max(h, 0.002);
    if (t > 30.0) { break; }
  }
  return 1.0;
}

fn soft_shadow(ro: vec3<f32>, rd: vec3<f32>, k: f32) -> f32 {
  var visibility = 1.0;
  var t = 0.02;
  let budget = trace_steps() / 3;
  for (var i = 0; i < budget; i++) {
    let h = traced_field(ro + rd * t);
    if (h < 0.001) { return 0.0; }
    visibility = min(visibility, k * h / t);
    t += max(h * 0.9, 0.0005);
    if (t > 30.0) { break; }
  }
  return clamp(visibility, 0.0, 1.0);
}

// Ambient occlusion, from the field itself.
//
// Four samples up the normal: where the field there is smaller than the
// distance stepped, other surface is close by — a corner, a bore, the
// underside of a deck — and the sky reaches it less. Four field reads per
// shaded pixel against a march of dozens, so it is close to free, and it is
// the one cue that says a cavity is a cavity rather than a dark face.
fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>) -> f32 {
  var occlusion = 0.0;
  var weight = 1.0;
  for (var i = 1; i <= 4; i++) {
    let h = 0.012 + 0.03 * f32(i);
    occlusion += max(h - traced_field(p + n * h), 0.0) * weight;
    weight *= 0.6;
  }
  return clamp(1.0 - AO_STRENGTH * occlusion, AO_FLOOR, 1.0);
}

// The viewport ground.
//
// \`u.bg_color\` is the ground colour in *linear radiance*: the caller has
// already inverted this shader's tone map and gamma, so a ray that hits
// nothing lands on exactly the colour the viewport declares.
//
// It is flat, and the flatness is the point. The old version was a gradient —
// ground at 0.45x, sky brightened to 1.8x and mixed toward a hard-coded blue
// — which a dark viewport needs, because there the environment is the only
// thing lighting a miss. Two things rule it out here. A light ground washes
// every face toward its own value, so the gradient's bright half erases the
// model; and the default projection is orthographic, where every primary ray
// shares one direction, so a direction-dependent environment collapses to a
// single colour that *changes as the camera orbits*. A ground you measure a
// field against cannot be a function of the camera.
fn environment_radiance(direction: vec3<f32>) -> vec3<f32> {
  return max(u.bg_color.xyz, vec3<f32>(0.001));
}

fn fresnel_schlick(cosine: f32, f0: vec3<f32>) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) *
    pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

/// Rotate a direction about the world's up axis (+Z).
fn rotate_about_up(direction: vec3<f32>, angle: f32) -> vec3<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec3<f32>(
    direction.x * c - direction.y * s,
    direction.x * s + direction.y * c,
    direction.z,
  );
}

/**
 * Sky above, ground below, keyed on the world's up axis.
 *
 * The viewport's own radiance is the sky; the ground is that radiance
 * darkened, which is what a light floor under a part actually does — it
 * bounces, but less than the dome above it. Both are achromatic, so this
 * lifts an unlit face without tinting it: the model's colour stays the
 * model's own, and a viridis field drawn on it is still readable as viridis.
 */
fn hemisphere_radiance(normal: vec3<f32>) -> vec3<f32> {
  let sky = environment_radiance(vec3<f32>(0.0, 0.0, 1.0));
  return mix(sky * GROUND_BOUNCE, sky, 0.5 + 0.5 * normal.z);
}

fn shade_material(
  position: vec3<f32>,
  normal: vec3<f32>,
  ray_direction: vec3<f32>,
) -> vec3<f32> {
  let base = material_base(position);
  let optics = material_optics(position);
  let base_color = clamp(base.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
  let roughness = clamp(base.w, 0.04, 1.0);
  let metallic = clamp(optics.x, 0.0, 1.0);
  let opacity = clamp(optics.y, 0.0, 1.0);
  let reflectivity = clamp(optics.w, 0.0, 1.0);
  let view_direction = -ray_direction;
  let light_direction = safe_normalize(u.light_dir.xyz);
  let half_vector = safe_normalize(view_direction + light_direction);
  let normal_dot_light = max(dot(normal, light_direction), 0.0);
  let normal_dot_view = max(dot(normal, view_direction), 0.0);
  let normal_dot_half = max(dot(normal, half_vector), 0.0);
  let view_dot_half = max(dot(view_direction, half_vector), 0.0);

  let alpha = roughness * roughness;
  let alpha_squared = alpha * alpha;
  let distribution_denominator =
    normal_dot_half * normal_dot_half * (alpha_squared - 1.0) + 1.0;
  let distribution = alpha_squared /
    max(PI * distribution_denominator * distribution_denominator, 1e-6);
  let geometry_k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
  let geometry_light = normal_dot_light /
    max(normal_dot_light * (1.0 - geometry_k) + geometry_k, 1e-6);
  let geometry_view = normal_dot_view /
    max(normal_dot_view * (1.0 - geometry_k) + geometry_k, 1e-6);
  let f0 = mix(vec3<f32>(0.04), base_color, metallic);
  let fresnel = fresnel_schlick(view_dot_half, f0);
  let specular = distribution * geometry_light * geometry_view * fresnel /
    max(4.0 * normal_dot_view * normal_dot_light, 1e-6);
  let diffuse =
    (vec3<f32>(1.0) - fresnel) * (1.0 - metallic) * base_color / PI;
  let shadow_origin = position + normal * 0.004;
  var visibility = 1.0;
  if (display_flag(DISPLAY_SHADOWS)) {
    if (display_flag(DISPLAY_HARD_SHADOWS)) {
      visibility = mix(HARD_SHADOW_FLOOR, 1.0, hard_shadow(shadow_origin, light_direction));
    } else {
      visibility = soft_shadow(shadow_origin, light_direction, 16.0);
    }
  }

  // Silhouette contour.
  //
  // On a dark ground a part reads because it is the bright thing; on a light
  // one it has to read by its edge, because the tone map pushes any lit face
  // and the background into the same shoulder. This darkens only the sliver
  // where the surface turns away from the eye — a drawn outline that follows
  // the geometry, costing no value anywhere on the interior.
  let facing = abs(dot(normal, view_direction));
  let contour = mix(0.16, 1.0, smoothstep(0.0, 0.30, facing));
  let occlusion = ambient_occlusion(shadow_origin, normal);

  // The cut face, darkened.
  //
  // Drafting hatches a section because the cut is a fiction — it is not a
  // surface the part has, and a reader has to be able to tell it from one
  // that is. A hatch cannot be drawn here: it is a screen-space pattern, and
  // this shader is also what the path tracer converges to and what the
  // thumbnailer renders at other sizes, so a fixed pitch would alias in one
  // and disappear in the other. A flat multiplier says the same thing at
  // every scale. 0.72 is the value that separates the cap from the darkest
  // real face at a grazing angle without turning it into a hole.
  let cut = select(1.0, SECTION_CAP_TONE, on_section_cap(position));

  // The rig's two fills, 120 degrees either side of the key about world up.
  // Declared here because both branches below need them: between the three
  // directions no normal is more than 60 degrees from a light, which is what
  // stops a face going dark just because the camera moved to it.
  let fill_a = rotate_about_up(light_direction, 2.0944);
  // The second fill is turned the other way *and* pushed below the horizon.
  //
  // A textbook three-point rig keeps every lamp above the subject, because a
  // photographer's subject is not inspected from underneath. A part is: the
  // orbit goes all the way round, and with all three lamps above, a face
  // pointing down received nothing but the ground half of the hemisphere.
  // That was the one view the rig did not fix — 56 % of the model's pixels
  // still below a readable level when the camera looked up at the board.
  let fill_b_azimuth = rotate_about_up(light_direction, -2.0944);
  let fill_b = normalize(vec3<f32>(fill_b_azimuth.xy, -abs(fill_b_azimuth.z)));
  let fill_incidence =
    max(dot(normal, fill_a), 0.0) * 0.34 + max(dot(normal, fill_b), 0.0) * 0.26;

  if (display_flag(DISPLAY_FLAT)) {
    // Flat shading: albedo lit only by incidence and shadowing — no specular
    // and no environment. The floor was 0.35 against a black viewport, where
    // the only risk was an unlit face disappearing; on paper the risk is the
    // opposite, a lit face washing into the ground. The range is quoted in
    // linear radiance and the tone map's shoulder eats most of the top of it:
    // 0.30..0.92 of a 0.85 albedo lands at sRGB 166..224 against a 230 ground,
    // which is the pale blob. 0.13..0.48 lands at 105..193 — a solid grey part
    // — and the contour closes the silhouette.
    // The key's share is shadowed; the fills' is not, for the same reason
    // they cast nothing. The window is unchanged at its ends — an unlit face
    // still lands at 0.13 and a fully lit one at 0.48 — but a face the key
    // has turned away from now sits inside it rather than on the floor.
    //
    // Under the X-Ray veil that window read flat: tops and sides of a part
    // landed within a few sRGB steps of each other, and the form went to the
    // skeleton drawn over it. So the key leads a little more — the fills
    // carry 0.8 of their share — and the window opens to 0.11..0.52, which
    // separates faces at different angles by roughly a third more without
    // moving the part's mean tone.
    //
    // Still too even in the shaded part: the deck, the walls and the skirt
    // sat within a few steps. Now the key carries three quarters of the
    // incidence, the fills half their share, and a quarter comes from which
    // way the face points against world up — the deck lighter, the
    // underside darker — over a window of 0.06..0.64. Occlusion darkens the
    // whole of it in corners and bores.
    let up = 0.5 + 0.5 * normal.z;
    let incidence = clamp(
      0.75 * normal_dot_light * visibility + 0.25 * up + fill_incidence * 0.5,
      0.0,
      1.0,
    );
    return base_color * mix(0.06, 0.64, incidence) * occlusion * contour * cut;
  }
  let intensity = max(u.light_dir.w, 1.0);
  // **White, not warm.** This used to be vec3(1.0, 0.92, 0.82), which tinted
  // every face of every part. Colour in this app is a reading — a material,
  // or a viridis field — and a light that pushes the whole model toward amber
  // competes with both. The rig is achromatic and the model's colour is the
  // model's own.
  let light_radiance = vec3<f32>(1.0) * intensity;
  let direct =
    (diffuse + specular) * light_radiance * normal_dot_light * visibility * KEY_GAIN;

  // Two fills, at a third and a fifth of the key.
  //
  // One directional light leaves a whole hemisphere of normals on ambient
  // alone, so orbiting to the far side of a part gave a flat dark shape —
  // which is what "objects are better visible from more angles" was about.
  // The fills sit 120 degrees either side of the key about the world's up
  // axis, so between the three of them no normal is more than 60 degrees from
  // a light. They are diffuse-only and cast nothing: a fill that threw its
  // own shadow would put a second, contradictory shadow under the part, and
  // the cost of the extra marches is the whole reason the key is the only
  // one that gets them.
  //
  // On the same footing as the key's diffuse (the Lambert 1/π), which they
  // once lacked: unnormalised, the two fills together matched the key, and
  // every face of a part came out the same grey whichever way it pointed.
  let fill = base_color * (1.0 - metallic) / PI * intensity * fill_incidence;

  // A hemisphere rather than a constant.
  //
  // \`environment_radiance\` is one flat colour, so ambient used to add the
  // same amount to every normal — light, but formless, and it could not tell
  // an upward face from a downward one. A sky/ground pair keyed on the
  // world's up axis costs one mix and gives the unlit side of a part its
  // shape back.
  //
  // Occluded, so a bore or an inside corner falls off where the sky cannot
  // reach it; the fills are shaded by the same factor.
  let ambient = base_color * hemisphere_radiance(normal) * 0.15 * occlusion;
  let reflected = environment_radiance(reflect(ray_direction, normal));
  let mirror = select(0.0, reflectivity, display_flag(DISPLAY_REFLECTIONS));
  let opaque = mix(direct + fill * occlusion + ambient, reflected, mirror) * contour * cut;
  let glass_fresnel = fresnel_schlick(
    max(dot(view_direction, normal), 0.0),
    vec3<f32>(0.04),
  );
  let transparent = mix(
    environment_radiance(ray_direction) * base_color,
    reflected,
    glass_fresnel,
  );
  return mix(transparent, opaque, opacity);
}

fn aces_tone_map(color: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp(
    (color * (a * color + vec3<f32>(b))) /
      (color * (c * color + vec3<f32>(d)) + vec3<f32>(e)),
    vec3<f32>(0.0),
    vec3<f32>(1.0),
  );
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

struct TraceResult {
  color    : vec3<f32>,
  position : vec3<f32>,
  hit      : bool,
  occludes : bool,
  /// Set on a pixel the slice owns: its colour is already display-encoded.
  sdf_view : bool,
};

// ── the SDF views ──────────────────────────────────────────────────────────
//
// A diverging ramp centred on zero: violet inside the solid, ochre outside it,
// the viewport's own paper at the crossing. Mirrored from
// \`frontend/src/viewer/sdfRamp.ts\`, which carries the measurements that chose
// the two hues and how far they sit from viridis and magma.
//
// These are *display* values, and everything else this shader produces is
// linear radiance that then goes through ACES and gamma. The slice is data,
// not light — a tone map applied to a ramp is a ramp you can no longer read a
// value off — so \`sdf_view_color\` returns the display value directly and
// \`trace_pixel\` marks the pixel so the tone map is skipped.
const SDF_INSIDE: vec3<f32> = vec3<f32>(0.3924, 0.0015, 0.9925);
const SDF_OUTSIDE: vec3<f32> = vec3<f32>(0.5398, 0.3384, 0.0033);
const SDF_CENTRE: vec3<f32> = vec3<f32>(0.902, 0.902, 0.914);
/// The achromatic ink both isoline families are drawn in.
const SDF_ISOLINE: vec3<f32> = vec3<f32>(0.12, 0.12, 0.13);

/// The ramp at a normalized signed value, t in [-1, 1]. See \`sdfRamp.ts\` for
/// why the interpolation is in sqrt(|t|) rather than in |t|.
fn sdf_ramp(t: f32) -> vec3<f32> {
  let clamped = clamp(t, -1.0, 1.0);
  let end = select(SDF_OUTSIDE, SDF_INSIDE, clamped < 0.0);
  return mix(SDF_CENTRE, end, sqrt(abs(clamped)));
}

/**
 * How dark this pixel's isoline ink is, 0…1.
 *
 * Screen-space derivatives turn "within half a line width of a multiple of
 * \`spacing\`" into a line that is one pixel wide whatever the zoom — the same
 * contract the floor grid holds itself to, which is why the spacing handed in
 * is the floor grid's own. A contour that thickened as you zoomed out would
 * stop being a contour and start being a band.
 */
fn sdf_isoline(value: f32, spacing: f32, width: f32) -> f32 {
  if (spacing <= 0.0) { return 0.0; }
  let scaled = value / spacing;
  let gradient = max(fwidth(scaled), 1e-6);
  let distance = abs(fract(scaled - 0.5) - 0.5) / gradient;
  // Screen-space level of detail, and the whole far field depends on it.
  //
  // \`gradient\` is how many intervals this pixel spans, so its reciprocal is
  // how many pixels one interval is worth. Out where the field runs fast —
  // away from the part, or at a shallow grazing angle — that falls below a
  // pixel, and a contour test sampled once per pixel then answers yes or no
  // essentially at random: the plane turns to black-and-white speckle. There
  // is no amount of anti-aliasing that fixes an interval finer than the
  // sampling; the only correct answer is to stop drawing that tier. It fades
  // out between two and six pixels per interval and the diverging ramp is
  // left to carry the value on its own, which it does without aliasing
  // because it is continuous.
  let pixels_per_interval = 1.0 / gradient;
  let lod = smoothstep(2.0, 6.0, pixels_per_interval);
  return (1.0 - smoothstep(0.0, width, distance)) * lod;
}

/// The single, heavier contour at one stated value — the zero level set on a
/// signed slice, |grad f| = 1 on a gradient slice.
fn sdf_level_line(value: f32, width: f32) -> f32 {
  let gradient = max(fwidth(value), 1e-6);
  return 1.0 - smoothstep(0.0, width, abs(value) / gradient);
}

/**
 * Linear camera depth, mapped bright-near to dark-far.
 *
 * The range is the *framed* depth rather than the clip planes: the near end
 * is half a frame height in front of the orbit target and the far end half a
 * frame behind it, so what fills the ramp is what fills the picture. Clip
 * planes would spend almost the whole ramp on empty space either side of the
 * part and leave the part itself a flat mid grey — the usual reason a z-pass
 * is unreadable. It follows the zoom, like the grid gain and the slice's own
 * contour interval, and the two ends are printed in the viewport readout.
 */
fn depth_tone(along_ray: f32) -> f32 {
  let orbit = length(u.camera_pos.xyz - u.camera_target.xyz);
  let frame = max(u.display.y, 1e-4);
  let near = max(orbit - 0.5 * frame, 1e-3);
  let far = orbit + 0.5 * frame;
  return 1.0 - clamp((along_ray - near) / max(far - near, 1e-6), 0.0, 1.0);
}

struct SliceHit {
  distance : f32,
  hit      : bool,
};

/**
 * Half-width of the cutting card, in world units.
 *
 * The plane is mathematically infinite; the thing drawn is not. A card of
 * stated size leaves the rest of the viewport for the faded solid, so the
 * slice sits *in* the scene and can be seen in relation to it — an infinite
 * plane fills the frame and there is nothing left to relate it to.
 *
 * Mirrors \`SDF_SLICE_RANGE\` in \`frontend/src/viewer/display.ts\`, which is also
 * the travel of the slice control, so the card sweeps a cube of its own size.
 */
const SDF_CARD_HALF: f32 = 2.0;

/// Where the primary ray meets the cutting card. A ray running along the plane,
/// arriving from behind the camera, or landing off the card's edge is a miss.
fn slice_intersection(ro: vec3<f32>, rd: vec3<f32>) -> SliceHit {
  let axis = i32(round(clamp(u.resolution.w, 0.0, 2.0)));
  var normal = vec3<f32>(0.0, 0.0, 0.0);
  normal[axis] = 1.0;
  let denominator = dot(rd, normal);
  if (abs(denominator) < 1e-5) { return SliceHit(0.0, false); }
  let distance = (u.camera_pos.w - dot(ro, normal)) / denominator;
  let point = ro + rd * distance;
  var on_card = distance > 0.0;
  for (var i = 0; i < 3; i++) {
    if (i != axis && abs(point[i]) > SDF_CARD_HALF) { on_card = false; }
  }
  return SliceHit(distance, on_card);
}

/**
 * The slice's colour at a world point.
 *
 * \`mode\` picks what is being drawn: the signed field itself, or |∇f|. Both are
 * normalized against something the viewport already states rather than against
 * a hidden constant — the field against half the framed height, so a full
 * ramp side is exactly half a screen of distance at the current gain, and the
 * gradient against ±0.5 about 1.0, so an exact SDF is the paper it is drawn on
 * and every departure from a metric field is coloured.
 */
fn sdf_view_color(point: vec3<f32>, mode: f32) -> vec3<f32> {
  let spacing = u.bg_color.w;
  let gradient = mode == SDF_VIEW_GRADIENT;
  // One scalar, two meanings, so the derivatives below are taken once and in
  // straight-line code: the signed field, or its own gradient magnitude.
  // Neither is clamped or folded — a negative distance is drawn exactly as a
  // positive one is, in the other hue, so the interior of a part is a graded
  // field running to its medial axis and not a flat fill.
  let step = max(spacing * 0.02, 1e-4);
  let value = select(scene_field(point), sdf_gradient_magnitude(point, step) - 1.0, gradient);

  // The ramp saturates two major divisions out, which is also where the minor
  // contours stop. One number governs both, so the hue and the line density
  // are telling the reader the same thing.
  let major = select(spacing, 0.1, gradient);
  let minor = major * 0.2;
  let scale = select(2.0 * spacing, 0.5, gradient);
  var color = sdf_ramp(value / scale);

  // Two tiers of contour.
  //
  // Uniform spacing at the grid rung is right out where the field is smooth
  // and far too coarse where it is interesting: everything worth reading off
  // a distance field happens within a division or two of the surface. So the
  // minor tier is a fifth of the major — the floor grid's own subdivision, so
  // a contour interval is always a stateable number — and it is faded out
  // across the band it covers rather than stopped at its edge, which would
  // draw a ring around nothing. The gradient view keeps one uniform family:
  // its interesting region is a *value*, 1.0, not a neighbourhood of a
  // surface, so a taper would have nothing to taper toward.
  let taper = select(
    1.0 - smoothstep(0.0, 2.0 * major, abs(value)),
    0.0,
    gradient,
  );
  color = mix(color, SDF_ISOLINE, sdf_isoline(value, minor, 0.75) * taper * 0.3);
  color = mix(
    color,
    SDF_ISOLINE,
    sdf_isoline(value, major, 0.9) * select(0.5, 0.35, gradient),
  );
  // And the one heavier line at the value that matters: the zero level set on
  // a signed slice — which is the part's own section outline — or |∇f| = 1 on
  // a gradient slice.
  return mix(color, SDF_ISOLINE, sdf_level_line(value, select(1.5, 1.2, gradient)));
}

fn trace_pixel(frag_xy: vec2<f32>) -> TraceResult {
  let res = u.resolution.xy;
  let uv = (frag_xy / res - 0.5) * vec2<f32>(res.x / res.y, -1.0);
  let ray = primary_ray(
    uv,
    u.camera_pos.xyz,
    u.camera_target.xyz,
    u.display.x,
    u.display.y,
  );
  let hide_solid = display_flag(DISPLAY_HIDE_SOLID);
  let distance = select(trace(ray.origin, ray.direction), -1.0, hide_solid);

  var result: TraceResult;
  result.color = environment_radiance(ray.direction);
  result.position = ray.origin;
  result.hit = distance >= 0.0;
  result.occludes = result.hit;
  result.sdf_view = false;

  let view_mode = u.resolution.z;
  if (result.hit) {
    let position = ray.origin + ray.direction * distance;
    let normal = sdf_normal(position);
    result.color = shade_material(position, normal, ray.direction);
    result.position = position;

    // The two surface views replace the shading outright. Both are data, not
    // light, so they skip the tone map — see \`resolve_color\`.
    if (view_mode == SDF_VIEW_NORMAL) {
      // The standard encoding, n * 0.5 + 0.5, in *world* axes: +X red, +Y
      // green, +Z blue, so a face pointing at the sky is the blue everyone
      // expects a normal map to be.
      result.color = normal * 0.5 + vec3<f32>(0.5);
      result.sdf_view = true;
    } else if (view_mode == SDF_VIEW_DEPTH) {
      result.color = vec3<f32>(depth_tone(distance));
      result.sdf_view = true;
    }

    let xray = clamp(u.display.w, 0.0, 1.0);
    if (xray > 0.0 && !sdf_view_shades_surface(view_mode)) {
      // Fade faces that point at the viewer and keep grazing angles solid, so
      // silhouettes and creases stay legible while interiors turn translucent.
      let facing = 1.0 - abs(dot(normal, ray.direction));
      // 0.78 at the floor, up from 0.62: the veil is mixed into every face
      // alike, so the further it reaches the more it flattens the shading
      // between them.
      let alpha = mix(0.78, 0.98, facing * facing);
      // On paper a face cannot fade toward the background: the ground is the
      // brightest thing on screen, so "transparent" and "absent" would be the
      // same pixel and an x-rayed solid disappears entirely (measured: sRGB
      // 222 of geometry against a 225 ground). It fades toward a veil instead
      // — the ground knocked down a fifth — and it fades less far, so the
      // ghost is a pale tint at sRGB ~208 with its grazing edges running down
      // to ~104. That is the same reading the dark viewport got from the
      // opposite direction: faint where you look through the part, strong
      // where the surface turns away.
      result.color = mix(
        environment_radiance(ray.direction) * 0.42,
        result.color,
        mix(1.0, alpha, xray),
      );
      // A contour where the surface turns away from the eye, in ink: the
      // ghost is then separated from the ground by an outline, not by a
      // fifth of lightness alone, and the skeleton drawn over it is the only
      // colour on it.
      let contour = smoothstep(0.82, 0.97, facing);
      result.color = mix(result.color, vec3<f32>(0.110, 0.094, 0.078), contour * 0.6 * xray);
      // An x-rayed solid still *is* somewhere, and it still writes its depth.
      //
      // This used to clear \`occludes\`, which made the depth buffer say the ray
      // missed — and the floor grid, which draws behind whatever the scene has
      // written, then came through the part. On the default (X-Ray) preset the
      // ground grid was visible straight through the heat sink, so the drawing
      // read as a grid laid over the model rather than a model standing on the
      // ground. Depth is the answer to "what is in front of what" and the
      // solid may not lie about it to buy an overlay a see-through.
      //
      // Construction geometry gets its see-through from its own end instead:
      // while the solid is x-rayed the overlay pipelines are built with
      // \`depthCompare: "always"\` (\`createOverlayPipelines\` in
      // \`viewer/pipelines.ts\`), which is what the gizmos have always used. The
      // grid is not construction geometry — it is the ground the part stands
      // on — so it keeps testing depth and is occluded by every silhouette.
    } else if (!sdf_view_shades_surface(view_mode)) {
      // The same ink contour on an opaque part, lighter: the part is then
      // parted from the ground by an outline in every preset, not by its
      // shading alone, which on paper can land on the ground's own value.
      let facing = 1.0 - abs(dot(normal, ray.direction));
      let contour = smoothstep(0.82, 0.97, facing);
      result.color = mix(result.color, vec3<f32>(0.110, 0.094, 0.078), contour * 0.45);
    }
  }

  // The slice, composited last so it can read the solid it stands in front of.
  //
  // Depth order is respected rather than assumed: the plane is drawn where it
  // is nearer than the surface, and the solid occludes it where it is not, so
  // the picture is a cut through the scene and not an overlay floating on it.
  // Everywhere the solid survives it is stepped back toward the ground, which
  // leaves the part legible as a silhouette without letting it compete with a
  // ramp the reader is about to take a value off.
  //
  // The shape of this block is dictated by WGSL's uniformity analysis, and it
  // is worth saying why. \`sdf_view_color\` takes screen-space derivatives — the
  // isolines are one pixel wide at every zoom, which is a \`fwidth\` — and a
  // derivative may only be taken where control flow is *uniform* across the
  // quad. \`u.resolution.z\` comes out of a uniform buffer, so branching on it
  // is uniform and the outer \`if\` is legal. Whether this pixel's ray reaches
  // the plane before it reaches the solid is not: it varies pixel to pixel.
  // So the colour is computed unconditionally inside the uniform branch and
  // only *chosen* inside the non-uniform one. wgpu's naga accepts the naive
  // nesting; Tint, which is what a browser actually compiles with, rejects it.
  // Written as one uniform if/else chain, and deliberately without an early
  // return: \`view_mode\` comes from a uniform buffer, so every arm here is
  // reached under uniform control flow, which is what lets the slice arm take
  // the screen-space derivatives its contours need. A \`return\` out of the
  // middle of it would end that, whatever the condition was.
  if (sdf_view_shades_surface(view_mode)) {
    // Nothing was hit, so there is no normal and no depth to report. Both
    // views state that rather than falling back on the lit background, which
    // would read as a surface: an absent normal is the encoding's own zero
    // vector, and an absent depth is past the far end of the ramp.
    if (!result.hit) {
      result.color = select(
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(depth_tone(1e6)),
        view_mode == SDF_VIEW_DEPTH,
      );
      result.sdf_view = true;
    }
  } else if (view_mode != SDF_VIEW_SOLID) {
    result.color = mix(environment_radiance(ray.direction), result.color, 0.28);
    let plane = slice_intersection(ray.origin, ray.direction);
    let point = ray.origin + ray.direction * max(plane.distance, 0.0);
    let slice_color = sdf_view_color(point, view_mode);
    // On the card, the field wins outright — the solid does not occlude it,
    // in front of the plane or behind it. That is what a section view is, and
    // it is the only way the *interior* of a part is visible at all: a plane
    // inside a solid is always behind that solid's own front surface, so depth
    // ordering hid exactly the half of the field the view exists to show.
    // Everywhere off the card the faded solid stays, which is what the card is
    // read against.
    if (plane.hit) {
      result.color = slice_color;
      result.position = point;
      result.hit = true;
      result.occludes = true;
      result.sdf_view = true;
    }
  }
  return result;
}

fn display_color(color: vec3<f32>) -> vec4<f32> {
  return vec4<f32>(pow(aces_tone_map(color), vec3<f32>(1.0 / 2.2)), 1.0);
}

/// Tone-map light, pass data through. See \`sdf_view_color\`.
fn resolve_color(result: TraceResult) -> vec4<f32> {
  if (result.sdf_view) { return vec4<f32>(result.color, 1.0); }
  return display_color(result.color);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  return resolve_color(trace_pixel(frag.xy));
}

struct DepthFragment {
  @location(0) color            : vec4<f32>,
  @builtin(frag_depth) depth    : f32,
};

// Same image as fs_main, plus scene depth so construction-tree overlays can be
// depth-tested against the rendered solid instead of floating on top of it.
@fragment
fn fs_main_depth(@builtin(position) frag: vec4<f32>) -> DepthFragment {
  let result = trace_pixel(frag.xy);
  var fragment: DepthFragment;
  fragment.color = resolve_color(result);
  fragment.depth = 1.0;
  if (result.occludes) {
    let clip = view.view_proj * vec4<f32>(result.position, 1.0);
    fragment.depth = clamp(clip.z / max(clip.w, 1e-6), 0.0, 1.0);
  }
  return fragment;
}
`,Qf=`__CADJOINT_SCENE_CODE__

const PI: f32 = 3.141592653589793;
const HIT_EPSILON: f32 = 0.0005;
const MAX_DISTANCE: f32 = 100.0;
const MAX_TRACE_STEPS: u32 = 160u;
const MAX_PATH_BOUNCES: u32 = 8u;
const MAX_SHADOW_SAMPLES: u32 = 4u;
const DISPLAY_SHADOWS: u32 = 1u;
const DISPLAY_REFLECTIONS: u32 = 2u;
const DISPLAY_FLAT: u32 = 4u;
const DISPLAY_HIDE_SOLID: u32 = 8u;
const DISPLAY_HARD_SHADOWS: u32 = 16u;
const HARD_SHADOW_FLOOR: f32 = 0.45;

struct Uniforms {
  resolution   : vec4<f32>,
  camera_pos   : vec4<f32>,
  camera_target: vec4<f32>,
  light_dir    : vec4<f32>,
  bg_color     : vec4<f32>,
  path_settings: vec4<f32>,
  display      : vec4<f32>,
};

struct TraceHit {
  distance: f32,
  hit: bool,
};

struct BsdfSample {
  direction: vec3<f32>,
  weight: vec3<f32>,
  valid: bool,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var previous_accumulation: texture_2d<f32>;


fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
  return v * inverseSqrt(max(dot(v, v), 1e-12));
}

struct CameraBasis {
  forward : vec3<f32>,
  right   : vec3<f32>,
  up      : vec3<f32>,
};

// Orthonormal camera frame. World up is +Z — the library is Z-up, since a
// SketchPlane's default normal is +Z and sketches lie on the XY floor — except
// when the view direction is almost parallel to it: looking straight down is
// exactly what the Top preset asks for, and cross(forward, +Z) is degenerate
// there.
fn camera_basis(camera: vec3<f32>, look_at: vec3<f32>) -> CameraBasis {
  var basis: CameraBasis;
  basis.forward = safe_normalize(look_at - camera);
  var reference = vec3<f32>(0.0, 0.0, 1.0);
  if (abs(basis.forward.z) > 0.999) {
    reference = vec3<f32>(0.0, 1.0, 0.0);
  }
  basis.right = safe_normalize(cross(basis.forward, reference));
  basis.up = cross(basis.right, basis.forward);
  return basis;
}

struct PrimaryRay {
  origin    : vec3<f32>,
  direction : vec3<f32>,
};

// Ray through a pixel, for either projection.
//
// \`uv\` is the screen coordinate in the viewer's convention:
//   uv = (frag.xy / resolution - 0.5) * vec2(aspect, -1)
//
// Perspective rays fan out from the camera with a half-height of
// FOV_SCALE / 2 per unit of depth; orthographic rays are parallel and offset
// across a viewport \`ortho_height\` world units tall, so a scene keeps its
// framing when \`ortho_height\` is the perspective frustum height at the orbit
// distance.
fn primary_ray(
  uv: vec2<f32>,
  camera: vec3<f32>,
  look_at: vec3<f32>,
  projection: f32,
  ortho_height: f32,
) -> PrimaryRay {
  let basis = camera_basis(camera, look_at);
  var ray: PrimaryRay;
  if (projection > 0.5) {
    ray.origin = camera + (uv.x * basis.right + uv.y * basis.up) * ortho_height;
    ray.direction = basis.forward;
  } else {
    ray.origin = camera;
    ray.direction = safe_normalize(
      basis.forward + 1.5 * (uv.x * basis.right + uv.y * basis.up),
    );
  }
  return ray;
}


fn display_flag(flag: u32) -> bool {
  return (u32(max(u.display.z, 0.0)) & flag) != 0u;
}

/**
 * The level set this tracer resolves, and how hard it is allowed to look.
 *
 * Both come from the same uniforms the preview shader reads (\`_webgpu.py\`
 * documents the packing), because the preview and the accumulated trace are
 * the same picture at two sample counts. Before this, the step budget was a
 * constant here and a different constant there — 160 against 96 — so "Ultra"
 * named two different marches on the two halves of one image.
 */
fn trace_steps() -> u32 {
  return select(u32(u.path_settings.w), MAX_TRACE_STEPS, u.path_settings.w < 1.0);
}

fn scene_field(p: vec3<f32>) -> f32 {
  return sdf(p) - u.camera_target.w;
}

fn pcg_hash(input: u32) -> u32 {
  let state = input * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn random_f32(state: ptr<function, u32>) -> f32 {
  *state = pcg_hash(*state);
  return f32(*state) * (1.0 / 4294967296.0);
}

fn signs_differ(a: f32, b: f32) -> bool {
  return (a < 0.0 && b >= 0.0) || (a > 0.0 && b <= 0.0);
}

fn refine_sign_crossing(
  origin: vec3<f32>,
  direction: vec3<f32>,
  near_distance: f32,
  far_distance: f32,
  near_surface_distance: f32,
) -> f32 {
  var near = near_distance;
  var far = far_distance;
  var near_sdf = near_surface_distance;
  for (var refinement = 0u; refinement < 7u; refinement += 1u) {
    let middle = 0.5 * (near + far);
    let middle_sdf = scene_field(origin + direction * middle);
    let same_side = (near_sdf < 0.0) == (middle_sdf < 0.0);
    if (same_side) {
      near = middle;
      near_sdf = middle_sdf;
    } else {
      far = middle;
    }
  }
  return 0.5 * (near + far);
}

fn trace_scene(origin: vec3<f32>, direction: vec3<f32>) -> TraceHit {
  var distance = 0.0;
  var surface_distance = scene_field(origin);
  let budget = trace_steps();
  for (var step = 0u; step < budget; step += 1u) {
    let advance = max(abs(surface_distance) * 0.9, HIT_EPSILON * 0.5);
    let next_distance = distance + advance;
    if (next_distance >= MAX_DISTANCE) {
      break;
    }
    let next_surface_distance = scene_field(origin + direction * next_distance);
    if (signs_differ(surface_distance, next_surface_distance)) {
      let refined_distance = refine_sign_crossing(
        origin,
        direction,
        distance,
        next_distance,
        surface_distance,
      );
      return TraceHit(refined_distance, true);
    }
    distance = next_distance;
    surface_distance = next_surface_distance;
  }
  return TraceHit(MAX_DISTANCE, false);
}

fn visible_to_directional_light(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
  var distance = 0.0;
  var surface_distance = scene_field(origin);
  if (surface_distance <= 0.0) {
    return 0.0;
  }
  let budget = trace_steps();
  for (var step = 0u; step < budget; step += 1u) {
    let advance = max(surface_distance * 0.9, HIT_EPSILON * 0.5);
    let next_distance = distance + advance;
    if (next_distance >= 50.0) {
      break;
    }
    let next_surface_distance = scene_field(origin + direction * next_distance);
    if (next_surface_distance <= 0.0) {
      return 0.0;
    }
    distance = next_distance;
    surface_distance = next_surface_distance;
  }
  return 1.0;
}

fn sdf_normal(position: vec3<f32>) -> vec3<f32> {
  let e = 0.00075;
  return safe_normalize(vec3<f32>(
    scene_field(position + vec3<f32>( e, 0.0, 0.0)) -
      scene_field(position + vec3<f32>(-e, 0.0, 0.0)),
    scene_field(position + vec3<f32>(0.0,  e, 0.0)) -
      scene_field(position + vec3<f32>(0.0, -e, 0.0)),
    scene_field(position + vec3<f32>(0.0, 0.0,  e)) -
      scene_field(position + vec3<f32>(0.0, 0.0, -e)),
  ));
}

// The viewport ground — see the identical function in \`_webgpu.py\` for why it
// is flat. Keep the two in sync: the preview and the accumulated path trace
// are the same picture at two sample counts, and a background that changes
// when the trace starts reads as a bug.
fn environment_radiance(direction: vec3<f32>) -> vec3<f32> {
  return max(u.bg_color.xyz, vec3<f32>(0.001));
}

fn orthonormal_basis(normal: vec3<f32>) -> mat3x3<f32> {
  let helper = select(
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(1.0, 0.0, 0.0),
    abs(normal.z) > 0.999,
  );
  let tangent = safe_normalize(cross(helper, normal));
  let bitangent = cross(normal, tangent);
  return mat3x3<f32>(tangent, bitangent, normal);
}

fn sample_cosine_hemisphere(
  normal: vec3<f32>,
  random_sample: vec2<f32>,
) -> vec3<f32> {
  let radius = sqrt(random_sample.x);
  let phi = 2.0 * PI * random_sample.y;
  let local = vec3<f32>(
    radius * cos(phi),
    radius * sin(phi),
    sqrt(max(0.0, 1.0 - random_sample.x)),
  );
  return safe_normalize(orthonormal_basis(normal) * local);
}

fn sample_sun_direction(
  center_direction: vec3<f32>,
  random_sample: vec2<f32>,
) -> vec3<f32> {
  let angular_radius = 0.025;
  let cos_theta = mix(cos(angular_radius), 1.0, random_sample.x);
  let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  let phi = 2.0 * PI * random_sample.y;
  let local = vec3<f32>(
    sin_theta * cos(phi),
    sin_theta * sin(phi),
    cos_theta,
  );
  return safe_normalize(orthonormal_basis(center_direction) * local);
}

fn sample_ggx_half_vector(
  normal: vec3<f32>,
  alpha: f32,
  random_sample: vec2<f32>,
) -> vec3<f32> {
  let alpha_squared = alpha * alpha;
  let phi = 2.0 * PI * random_sample.x;
  let cos_theta = sqrt(
    max(0.0, (1.0 - random_sample.y) /
      (1.0 + (alpha_squared - 1.0) * random_sample.y)),
  );
  let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  let local = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
  return safe_normalize(orthonormal_basis(normal) * local);
}

fn ggx_distribution(alpha: f32, normal_dot_half: f32) -> f32 {
  let alpha_squared = alpha * alpha;
  let denominator =
    normal_dot_half * normal_dot_half * (alpha_squared - 1.0) + 1.0;
  return alpha_squared / max(PI * denominator * denominator, 1e-7);
}

fn geometry_schlick(normal_dot_direction: f32, roughness: f32) -> f32 {
  let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
  return normal_dot_direction /
    max(normal_dot_direction * (1.0 - k) + k, 1e-6);
}

fn fresnel_schlick_rgb(cosine: f32, f0: vec3<f32>) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_dielectric(cosine: f32, ior: f32) -> f32 {
  let r0_base = (1.0 - ior) / (1.0 + ior);
  let r0 = r0_base * r0_base;
  return r0 + (1.0 - r0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

fn evaluate_opaque_bsdf(
  base_color: vec3<f32>,
  roughness: f32,
  metallic: f32,
  normal: vec3<f32>,
  outgoing: vec3<f32>,
  incoming: vec3<f32>,
) -> vec3<f32> {
  let normal_dot_incoming = max(dot(normal, incoming), 0.0);
  let normal_dot_outgoing = max(dot(normal, outgoing), 0.0);
  if (normal_dot_incoming <= 0.0 || normal_dot_outgoing <= 0.0) {
    return vec3<f32>(0.0);
  }

  let half_vector = safe_normalize(outgoing + incoming);
  let normal_dot_half = max(dot(normal, half_vector), 0.0);
  let outgoing_dot_half = max(dot(outgoing, half_vector), 0.0);
  let perceptual_roughness = clamp(roughness, 0.04, 1.0);
  let alpha = perceptual_roughness * perceptual_roughness;
  let distribution = ggx_distribution(alpha, normal_dot_half);
  let geometry =
    geometry_schlick(normal_dot_incoming, perceptual_roughness) *
    geometry_schlick(normal_dot_outgoing, perceptual_roughness);
  let f0 = mix(vec3<f32>(0.04), base_color, metallic);
  let fresnel = fresnel_schlick_rgb(outgoing_dot_half, f0);
  let specular = distribution * geometry * fresnel /
    max(4.0 * normal_dot_incoming * normal_dot_outgoing, 1e-6);
  let diffuse =
    (vec3<f32>(1.0) - fresnel) * (1.0 - metallic) * base_color / PI;
  return diffuse + specular;
}

fn opaque_bsdf_pdf(
  roughness: f32,
  metallic: f32,
  normal: vec3<f32>,
  outgoing: vec3<f32>,
  incoming: vec3<f32>,
) -> f32 {
  let normal_dot_incoming = max(dot(normal, incoming), 0.0);
  if (normal_dot_incoming <= 0.0) {
    return 0.0;
  }
  let specular_probability = mix(0.25, 0.85, metallic);
  let diffuse_pdf = normal_dot_incoming / PI;
  let half_vector = safe_normalize(outgoing + incoming);
  let normal_dot_half = max(dot(normal, half_vector), 0.0);
  let outgoing_dot_half = max(abs(dot(outgoing, half_vector)), 1e-6);
  let alpha = clamp(roughness, 0.04, 1.0);
  let distribution = ggx_distribution(alpha * alpha, normal_dot_half);
  let specular_pdf = distribution * normal_dot_half / (4.0 * outgoing_dot_half);
  return mix(diffuse_pdf, specular_pdf, specular_probability);
}

fn sample_opaque_bsdf(
  base_color: vec3<f32>,
  roughness: f32,
  metallic: f32,
  normal: vec3<f32>,
  outgoing: vec3<f32>,
  random_lobe: f32,
  random_sample: vec2<f32>,
) -> BsdfSample {
  let specular_probability = mix(0.25, 0.85, metallic);
  var incoming: vec3<f32>;
  if (random_lobe < specular_probability) {
    let alpha = clamp(roughness, 0.04, 1.0);
    let half_vector = sample_ggx_half_vector(
      normal,
      alpha * alpha,
      random_sample,
    );
    incoming = reflect(-outgoing, half_vector);
  } else {
    incoming = sample_cosine_hemisphere(normal, random_sample);
  }

  let normal_dot_incoming = max(dot(normal, incoming), 0.0);
  let pdf = opaque_bsdf_pdf(
    roughness,
    metallic,
    normal,
    outgoing,
    incoming,
  );
  if (normal_dot_incoming <= 0.0 || pdf <= 1e-7) {
    return BsdfSample(incoming, vec3<f32>(0.0), false);
  }
  let bsdf = evaluate_opaque_bsdf(
    base_color,
    roughness,
    metallic,
    normal,
    outgoing,
    incoming,
  );
  return BsdfSample(incoming, bsdf * normal_dot_incoming / pdf, true);
}

fn trace_path(
  ray_origin: vec3<f32>,
  ray_direction: vec3<f32>,
  random_state: ptr<function, u32>,
) -> vec3<f32> {
  var origin = ray_origin;
  var direction = ray_direction;
  var radiance = vec3<f32>(0.0);
  var throughput = vec3<f32>(1.0);
  var eta_scale = 1.0;
  let configured_bounces = min(u32(u.path_settings.y), MAX_PATH_BOUNCES);
  var configured_shadow_samples = clamp(
    u32(u.path_settings.z),
    1u,
    MAX_SHADOW_SAMPLES,
  );
  let shadows_enabled = display_flag(DISPLAY_SHADOWS);
  let hard_shadows = display_flag(DISPLAY_HARD_SHADOWS);
  if (!shadows_enabled || hard_shadows) {
    configured_shadow_samples = 1u;
  }

  for (var bounce = 0u; bounce < MAX_PATH_BOUNCES; bounce += 1u) {
    if (bounce >= configured_bounces) {
      break;
    }

    let intersection = trace_scene(origin, direction);
    if (!intersection.hit) {
      // Backplate and light dome are the same colour but not the same
      // intensity. A camera ray that misses has to land on exactly the paper
      // the viewport declares; a *bounce* ray that misses is the studio's
      // fill light, and at paper's own radiance a white part lit by a white
      // dome converges on the ground and the silhouette dissolves. The dome
      // is therefore half the backplate — the standard backplate/HDRI split,
      // and the only knob that keeps a light part legible on a light ground
      // without touching the BSDF.
      let dome = select(0.5, 1.0, bounce == 0u);
      radiance += throughput * environment_radiance(direction) * dome;
      break;
    }

    let position = origin + direction * intersection.distance;
    let outward_normal = sdf_normal(position);
    let front_face = dot(direction, outward_normal) < 0.0;
    let normal = select(-outward_normal, outward_normal, front_face);
    let outgoing = -direction;
    let base = material_base(position);
    let optics = material_optics(position);
    let base_color = clamp(base.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let roughness = clamp(base.w, 0.04, 1.0);
    let metallic = clamp(optics.x, 0.0, 1.0);
    let xray = clamp(u.display.w, 0.0, 1.0);
    let facing = 1.0 - abs(dot(normal, direction));
    let xray_alpha = mix(1.0, mix(0.12, 0.95, facing * facing), xray);
    let opacity = clamp(optics.y, 0.0, 1.0) * xray_alpha;
    let ior = max(optics.z, 1.0001);
    let reflectivity = select(
      0.0,
      clamp(optics.w, 0.0, 1.0),
      display_flag(DISPLAY_REFLECTIONS),
    );

    if (display_flag(DISPLAY_FLAT)) {
      let light_direction = safe_normalize(u.light_dir.xyz);
      let normal_dot_light = max(dot(normal, light_direction), 0.0);
      var visibility = 1.0;
      if (shadows_enabled) {
        visibility = visible_to_directional_light(
          position + normal * (HIT_EPSILON * 6.0),
          light_direction,
        );
        if (hard_shadows) {
          visibility = mix(HARD_SHADOW_FLOOR, 1.0, visibility);
        }
      }
      let flat_color =
        base_color * mix(0.35, 1.0, normal_dot_light * visibility);
      return mix(environment_radiance(direction), flat_color, opacity);
    }

    if (
      opacity > 0.0 &&
      reflectivity < 1.0
    ) {
      let light_radiance =
        vec3<f32>(1.0, 0.92, 0.82) * max(u.light_dir.w, 0.0);
      var direct_lighting = vec3<f32>(0.0);
      for (
        var shadow_sample = 0u;
        shadow_sample < MAX_SHADOW_SAMPLES;
        shadow_sample += 1u
      ) {
        if (shadow_sample >= configured_shadow_samples) {
          break;
        }
        var light_direction = safe_normalize(u.light_dir.xyz);
        if (shadows_enabled && !hard_shadows) {
          light_direction = sample_sun_direction(
            light_direction,
            vec2<f32>(random_f32(random_state), random_f32(random_state)),
          );
        }
        let normal_dot_light = max(dot(normal, light_direction), 0.0);
        if (normal_dot_light > 0.0) {
          var visibility = 1.0;
          if (shadows_enabled) {
            visibility = visible_to_directional_light(
              position + normal * (HIT_EPSILON * 6.0),
              light_direction,
            );
            if (hard_shadows) {
              visibility = mix(HARD_SHADOW_FLOOR, 1.0, visibility);
            }
          }
          let direct_bsdf = evaluate_opaque_bsdf(
            base_color,
            roughness,
            metallic,
            normal,
            outgoing,
            light_direction,
          );
          direct_lighting +=
            direct_bsdf * light_radiance * normal_dot_light * visibility;
        }
      }
      radiance += throughput * opacity * (1.0 - reflectivity) *
        direct_lighting / f32(configured_shadow_samples);
    }

    let transparent_event = random_f32(random_state) >= opacity;
    if (transparent_event) {
      let cosine = max(dot(outgoing, normal), 0.0);
      let fresnel = fresnel_schlick_dielectric(cosine, ior);
      let eta = select(ior, 1.0 / ior, front_face);
      let transmitted_sine_squared =
        eta * eta * max(0.0, 1.0 - cosine * cosine);
      let total_internal_reflection = transmitted_sine_squared >= 1.0;
      if (
        total_internal_reflection ||
        random_f32(random_state) < fresnel
      ) {
        direction = reflect(direction, normal);
      } else {
        direction = safe_normalize(refract(direction, normal, eta));
        throughput *= eta * eta;
        eta_scale /= eta * eta;
        if (!front_face) {
          throughput *= base_color;
        }
      }
    } else {
      let mirror_event = random_f32(random_state) < reflectivity;
      if (mirror_event) {
        direction = reflect(direction, normal);
        throughput *= mix(vec3<f32>(1.0), base_color, metallic);
      } else {
        let sample = sample_opaque_bsdf(
          base_color,
          roughness,
          metallic,
          normal,
          outgoing,
          random_f32(random_state),
          vec2<f32>(random_f32(random_state), random_f32(random_state)),
        );
        if (!sample.valid) {
          break;
        }
        direction = sample.direction;
        throughput *= sample.weight;
      }
    }

    direction = safe_normalize(direction);
    let offset_sign = select(-1.0, 1.0, dot(direction, normal) >= 0.0);
    origin = position + normal * (HIT_EPSILON * 6.0 * offset_sign);

    if (bounce >= 2u) {
      let roulette_throughput = throughput * eta_scale;
      let survival_probability = clamp(
        max(
          roulette_throughput.x,
          max(roulette_throughput.y, roulette_throughput.z),
        ),
        0.05,
        0.95,
      );
      if (random_f32(random_state) >= survival_probability) {
        break;
      }
      throughput /= survival_probability;
    }
  }
  return max(radiance, vec3<f32>(0.0));
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
  let positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_path_trace(
  @builtin(position) fragment: vec4<f32>,
) -> @location(0) vec4<f32> {
  let pixel = vec2<i32>(fragment.xy);
  let sample_index = u32(u.path_settings.x);
  var random_state = pcg_hash(
    (u32(pixel.x) * 1973u) ^
    (u32(pixel.y) * 9277u) ^
    (sample_index * 26699u) ^
    0x68bc21ebu
  );
  let jitter = vec2<f32>(
    random_f32(&random_state) - 0.5,
    random_f32(&random_state) - 0.5,
  );

  let resolution = u.resolution.xy;
  let uv = ((fragment.xy + jitter) / resolution - 0.5) *
    vec2<f32>(resolution.x / resolution.y, -1.0);
  let ray = primary_ray(
    uv,
    u.camera_pos.xyz,
    u.camera_target.xyz,
    u.display.x,
    u.display.y,
  );
  let camera = ray.origin;
  let ray_direction = ray.direction;
  var sample_radiance = environment_radiance(ray_direction);
  if (!display_flag(DISPLAY_HIDE_SOLID)) {
    sample_radiance = trace_path(camera, ray_direction, &random_state);
  }

  let previous = textureLoad(previous_accumulation, pixel, 0).xyz;
  let sample_count = f32(sample_index);
  let accumulated =
    (previous * sample_count + sample_radiance) / (sample_count + 1.0);
  return vec4<f32>(accumulated, 1.0);
}
`,ep=`
@group(0) @binding(0) var accumulated_radiance: texture_2d<f32>;

fn aces_tone_map(color: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp(
    (color * (a * color + vec3<f32>(b))) /
      (color * (c * color + vec3<f32>(d)) + vec3<f32>(e)),
    vec3<f32>(0.0),
    vec3<f32>(1.0),
  );
}

@vertex
fn vs_present(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
  let positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_present(@builtin(position) fragment: vec4<f32>) -> @location(0) vec4<f32> {
  let pixel = vec2<i32>(fragment.xy);
  let linear_color = textureLoad(accumulated_radiance, pixel, 0).xyz;
  let display_color = pow(
    aces_tone_map(linear_color),
    vec3<f32>(1.0 / 2.2),
  );
  return vec4<f32>(display_color, 1.0);
}
`;function tp(e){const t=e.includes("fn material_base("),n=e.includes("fn material_optics(");if(t!==n)throw new Error("Scene source must define both material WGSL functions or neither");return t?e:`${e}

${Zf}`}function np(e){if(e.includes(Mi))throw new Error(`SDF source cannot contain the reserved marker '${Mi}'`);const t=tp(e);return Jf.replace(Mi,()=>t)}function ip(e){if(e.includes(Ci))throw new Error(`Scene source cannot contain the reserved marker '${Ci}'`);return Qf.replace(Ci,()=>e)}function rp(e,t=!0){const n=Kf(e,{uniforms:!0,culling:t}),i=new Set(n.derived),r=n.parameters.map(([s,o,l],c)=>{const d={name:s,offset:c*yn,components:o,value:l.map(u=>Number.isFinite(u)?u:null),free:!i.has(s)},h=n.depends[s];return h?.length&&(d.depends=h),d}),a=r.length;return{wgsl:n.wgsl,program:{group:Ct.group,binding:Ct.binding,buffer_bytes:(a+Ct.reservedSlots)*yn,nan_offset:a*yn,cull_margin_offset:(a+1)*yn,parameters:r}}}const yn=Ct.slotBytes;async function ap(e){const t=await crypto.subtle.digest("SHA-256",new TextEncoder().encode(e));return[...new Uint8Array(t)].map(n=>n.toString(16).padStart(2,"0")).join("")}async function sp(e,t=!0){const{wgsl:n,program:i}=rp(e,t),r=np(n),a=ip(n);return{sdf:n,shader:r,scene_wgsl:n,preview_shader:r,path_shader:a,present_shader:ep,program:i,shader_hash:await ap(`${r}\0${a}`)}}function op(e){const t=Uh(),n=t.presets.find(I=>I.id===t.activeId)??t.presets[0],[i,r]=T(n.pathTracing),[a,s]=T(n.quality),[o,l]=T({...n.display}),[c,d]=T(t.presets),[h,u]=T(t.activeId),[p,f]=T("iso");e.display={...n.display},e.quality=Ze[n.quality],e.pathTracing=n.pathTracing;const m=I=>{const A={...o(),...I};l(A),e.display=A,e.invalidate()},w=I=>{const A=Ze[I];A&&(s(I),e.quality=A,e.invalidate())},x=I=>{r(I),e.pathTracing=I,e.invalidate()};return{display:o,presets:c,selectedPreset:h,pathTracing:i,quality:a,viewPreset:p,applyDisplay:m,applyQuality:w,applyPathTracing:x,activateRenderPreset:I=>{const A=c().find(B=>B.id===I);A&&(u(I),m(A.display),w(A.quality),x(A.pathTracing),Si({presets:c(),activeId:I}))},saveRenderPreset:I=>{const A=c().map(B=>B.id===I?{...B,pathTracing:i(),quality:a(),display:{...o()}}:B);d(A),u(I),Si({presets:A,activeId:I})},resetRenderPreset:I=>{const A=Jn.find(N=>N.id===I);if(!A)return;const B=c().map(N=>N.id===I?{...A,display:{...A.display}}:N);d(B),u(I),m(A.display),w(A.quality),x(A.pathTracing),Si({presets:B,activeId:I})},applyPreset:I=>{f(I),e.applyViewPreset(I),l({...e.display}),xr({yaw:e.camera.yaw,pitch:e.camera.pitch})}}}class dt{constructor(t,n){this.label=t,this.stride=n}buffer=null;count=0;capacity=0;write(t,n){if(this.count=n.length/(this.stride/4),n.length===0)return;const i=n.byteLength;if(!this.buffer||this.capacity<i){this.buffer?.destroy();const r=Math.max(i,this.stride*64);this.buffer=t.createBuffer({label:this.label,size:r,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),this.capacity=r}t.queue.writeBuffer(this.buffer,0,n)}destroy(){this.buffer?.destroy(),this.buffer=null,this.capacity=0,this.count=0}}function lp(e){let t=0;for(let n=0,i=e.length-1;n<e.length;i=n++)t+=(e[i][0]-e[n][0])*(e[i][1]+e[n][1]);return t/2}const Bn=(e,t,n)=>(t[0]-e[0])*(n[1]-e[1])-(t[1]-e[1])*(n[0]-e[0]);function cp(e,t,n,i){const r=Bn(t,n,e),a=Bn(n,i,e),s=Bn(i,t,e),o=r<0||a<0||s<0,l=r>0||a>0||s>0;return!(o&&l)}function dp(e){const t=e.length;if(t<3)return[];const n=lp(e)>0,i=[];for(let s=0;s<t;s++)i.push(n?s:t-1-s);const r=[];let a=i.length*i.length;for(;i.length>3&&a-- >0;){let s=!1;for(let o=0;o<i.length;o++){const l=i[(o+i.length-1)%i.length],c=i[o],d=i[(o+1)%i.length],h=e[l],u=e[c],p=e[d];if(!(Bn(h,u,p)<=0||i.some(m=>m!==l&&m!==c&&m!==d&&cp(e[m],h,u,p)))){r.push(l,c,d),i.splice(o,1),s=!0;break}}if(!s)break}return i.length===3&&r.push(i[0],i[1],i[2]),r}const an=40,Mr=36,Fn=44,Ks=28,Zs=40,oe=Object.fromEntries(Object.keys(Za).map(e=>[e,rl(e)])),Ti=le(se.ink),vt=e=>[...le(se[e]),1],up={edge:vt(mt.rest),edgeHover:vt(mt.hover),edgeSelected:vt(mt.selected),edgeLocked:vt(mt.locked),handleHover:vt(mt.selected),handleSelected:vt(mt.selected),handleLocked:vt(mt.locked),held:vt(mt.rest)},hp=[...le(se[ct.rest]),1],fp=[...le(se.param),1],Oi={free:0,held:1,selected:2};function pp(e,t){const n=e.vertices[t];return n&&!n.free?!0:e.constraints.some(i=>i.kind==="fixed"&&i.vertices.includes(t))}function mp(e,t,n,i=!1,r=new Set){const a=[],s=[],o=i?up:{edge:oe.edge,edgeHover:oe.edgeHover,edgeSelected:oe.edgeSelected,edgeLocked:oe.edgeLocked,handleHover:oe.handleHover,handleSelected:oe.handleSelected,handleLocked:oe.handleLocked,held:hp};for(const l of e){const c=t?.nodeId===l.id||r.has(l.id),d=n?.nodeId===l.id&&n.vertexIndex===null,h=c?o.edgeSelected:d?o.edgeHover:l.editable?o.edge:o.edgeLocked;for(const[u,p]of l.edges)a.push(u[0],u[1],u[2],p[0],p[1],p[2],...h);if(l.editable)for(let u=0;u<l.vertices.length;u++){const p=t?.nodeId===l.id&&t.vertexIndex===u,f=n?.nodeId===l.id&&n.vertexIndex===u,m=pp(l,u),w=p?Oi.selected:m?Oi.held:Oi.free,x=p?o.handleSelected:m?f?o.handleHover:o.held:fp,y=l.vertices[u].world;s.push(y[0],y[1],y[2],...x,p||f?1:0,w)}}return{edges:a,handles:s}}function gp(e){const t=[],n=[];if(!e||e.polygon.length<3)return{fill:t,outline:n};const i=e.usable?oe.faceFill:oe.faceFillLocked,r=e.usable?oe.faceOutline:oe.faceOutlineLocked,a=e.polygon.map(o=>{const l=[o[0]-e.origin[0],o[1]-e.origin[1],o[2]-e.origin[2]];return[l[0]*e.xAxis[0]+l[1]*e.xAxis[1]+l[2]*e.xAxis[2],l[0]*e.yAxis[0]+l[1]*e.yAxis[1]+l[2]*e.yAxis[2]]});for(const o of dp(a)){const l=e.polygon[o];t.push(l[0],l[1],l[2],...i)}const s=e.polygon.length;for(let o=0;o<s;o++){const l=e.polygon[o],c=e.polygon[(o+1)%s];n.push(l[0],l[1],l[2],c[0],c[1],c[2],...r)}return{fill:t,outline:n}}function lt(e,t,n,i){e.push(t[0],t[1],t[2],n[0],n[1],n[2],...i)}function vp(e,t,n,i){const r=[],a=[],s=(o,l,c)=>{const[d,h,u,p]=o.corners;for(const A of[d,h,u,d,u,p])r.push(A[0],A[1],A[2],...l);lt(a,d,h,c),lt(a,h,u,c),lt(a,u,p,c),lt(a,p,d,c);const f=Rl(o),{origin:m,u:w,v:x,normal:y}=o,_=(A,B)=>[m[0]+A[0]*B,m[1]+A[1]*B,m[2]+A[2]*B];lt(a,_(w,-f),_(w,f),c),lt(a,_(x,-f),_(x,f),c);const $=Ll(o);lt(a,m,$,c);const E=f*.8,I=[$[0]-y[0]*E,$[1]-y[1]*E,$[2]-y[2]*E];lt(a,$,[I[0]+w[0]*E,I[1]+w[1]*E,I[2]+w[2]*E],c),lt(a,$,[I[0]-w[0]*E,I[1]-w[1]*E,I[2]-w[2]*E],c)};for(const o of e){const l=t?.nodeId===o.nodeId&&t.vertexIndex===null,c=l&&t?.part==="plane",d=n?.nodeId===o.nodeId&&n.vertexIndex===null&&n.part==="plane";o.derived&&!c&&!d?s(o,oe.planeFillLocked,oe.planeOutlineLocked):c?s(o,oe.planeFillSelected,oe.planeOutlineSelected):d||l?s(o,oe.planeFillHover,oe.planeOutlineHover):s(o,oe.planeFill,oe.planeOutline)}return i&&s(i,oe.planeFillPreview,oe.planeOutlinePreview),{fill:r,outline:a}}function bp(e,t,n,i){const r=[];for(const a of Qa(e,t,n)){const s=ol[a.axis],o=i===a.axis,l=o?[s[0]+(Ti[0]-s[0])*.3,s[1]+(Ti[1]-s[1])*.3,s[2]+(Ti[2]-s[2])*.3,1]:[s[0],s[1],s[2],.98],c=n==="rotate"?a.edges:a.edges.slice(0,1);for(const[d,h]of c)r.push(d[0],d[1],d[2],h[0],h[1],h[2],...l,o?1:0)}return r}function yp(e){const t=[];if(!e)return t;for(const[n,i]of e.wire)t.push(n[0],n[1],n[2],i[0],i[1],i[2],...oe.meshWire);for(const[n,i]of e.sharp)t.push(n[0],n[1],n[2],i[0],i[1],i[2],...oe.meshSharp);return t}class wp{pipelines=null;faces=new dt("B-rep faces",Zs);edges=new dt("B-rep edges",an);vertices=new dt("B-rep vertices",Mr);attach(t){this.pipelines=t}get hasFaces(){return this.faces.count>0}set(t,n){const i=new Float32Array(0);this.faces.write(t,n?.faces??i),this.edges.write(t,n?.edges??i),this.vertices.write(t,n?.vertices??i)}draw(t){const n=this.pipelines;!n||!this.faces.buffer||(t.setPipeline(n.brepFacePipeline),t.setBindGroup(0,n.overlayBindGroup),t.setVertexBuffer(0,this.faces.buffer),t.draw(this.faces.count),this.edges.count&&this.edges.buffer&&(t.setPipeline(n.edgePipeline.tested),t.setBindGroup(0,n.meshOverlayBindGroup),t.setVertexBuffer(0,this.edges.buffer),t.draw(6,this.edges.count)),this.vertices.count&&this.vertices.buffer&&(t.setPipeline(n.handlePipeline.tested),t.setBindGroup(0,n.overlayBindGroup),t.setVertexBuffer(0,this.vertices.buffer),t.draw(6,this.vertices.count)))}destroy(){this.faces.destroy(),this.edges.destroy(),this.vertices.destroy()}}const wn={r:_t[0],g:_t[1],b:_t[2],a:1};function Li(e){const t=Math.pow(e,2.2),n=2.51-2.43*t,i=.03-.59*t,r=-.14*t;return(-i+Math.sqrt(i*i-4*n*r))/(2*n)}const Di=[Li(_t[0]),Li(_t[1]),Li(_t[2])],_p=1.5,Sa=.004,xp=2.4,kp=.8,Sp=7.5,$a={lineWidth:1,axisWidth:1.6},Pa=12,$p=.4;function Ea(e,t,n){return Promise.race([e,new Promise((i,r)=>setTimeout(()=>r(new Error(`${n} timed out after ${t} ms.`)),t))])}class Pp{gizmoMode="translate";gizmoAxis=null;pipelines=null;profiles=[];planes=[];selection=null;hover=null;focusGroup=new Set;meshEdges=null;planePreview=null;faceHighlight=null;visibleGizmoMode="translate";edges=new dt("overlay edges",an);handles=new dt("overlay handles",Mr);gizmo=new dt("transform gizmo",Fn);faceFill=new dt("face highlight fill",Ks);faceOutline=new dt("face highlight outline",an);meshEdgeSegments=new dt("mesh edges",an);meshWireCount=0;meshSharpCount=0;attach(t){this.pipelines=t}get hasSkeletonEdges(){return this.edges.count>0}setConstruction(t,n,i,r){this.profiles=t,this.selection=n,this.hover=i,this.planes=r}setFocusGroup(t){return t.size===0&&this.focusGroup.size===0?!1:(this.focusGroup=t,!0)}setMeshEdges(t){this.meshEdges=t}setPlanePreview(t){return this.planePreview===t?!1:(this.planePreview=t,!0)}setFaceHighlight(t){return this.faceHighlight===t?!1:(this.faceHighlight=t,!0)}gizmoModeFor(t){return this.gizmoMode==="rotate"&&!t.transform?.canRotate||this.gizmoMode==="scale"&&t.kind==="profile"?"translate":this.gizmoMode}gizmoTarget(){return gl(this.selection,this.profiles)}upload(t,n,i){const{edges:r,handles:a}=mp(this.profiles,this.selection,this.hover,n,this.focusGroup),s=this.gizmoTarget();let o=[];s&&(this.visibleGizmoMode=this.gizmoModeFor(s.node),o=bp(s.origin,Ui(i,s.origin),this.visibleGizmoMode,this.gizmoAxis));const l=yp(this.meshEdges),c=gp(this.faceHighlight),d=vp(this.planes,this.selection,this.hover,this.planePreview),h=d.fill.concat(c.fill),u=d.outline.concat(c.outline);this.faceFill.write(t,new Float32Array(h)),this.faceOutline.write(t,new Float32Array(u)),this.meshWireCount=this.meshEdges?.wire.length??0,this.meshSharpCount=this.meshEdges?.sharp.length??0,this.meshEdgeSegments.write(t,new Float32Array(l)),this.edges.write(t,new Float32Array(r)),this.handles.write(t,new Float32Array(a)),this.gizmo.write(t,new Float32Array(o))}draw(t,n,i){const r=this.pipelines;if(!r)return;const{edgePipeline:a,handlePipeline:s,facePipeline:o,overlayBindGroup:l}=r,c=i?f=>f.seen:f=>f.tested,d=!!(n.showOverlays&&n.showSketches&&this.edges.count&&this.edges.buffer),h=!!(n.showOverlays&&n.showSketches&&this.handles.count&&this.handles.buffer),u=n.showMeshWireframe&&this.meshWireCount>0,p=n.showMeshEdges&&this.meshSharpCount>0;if((u||p)&&this.meshEdgeSegments.buffer&&(t.setPipeline(c(a)),t.setBindGroup(0,r.meshOverlayBindGroup),t.setVertexBuffer(0,this.meshEdgeSegments.buffer),u&&t.draw(6,this.meshWireCount,0,0),p&&t.draw(6,this.meshSharpCount,0,this.meshWireCount),t.setBindGroup(0,l)),!!n.showOverlays&&(i&&(d&&a.hidden&&(t.setPipeline(a.hidden),t.setBindGroup(0,l),t.setVertexBuffer(0,this.edges.buffer),t.draw(6,this.edges.count)),h&&s.hidden&&(t.setPipeline(s.hidden),t.setBindGroup(0,l),t.setVertexBuffer(0,this.handles.buffer),t.draw(6,this.handles.count))),this.faceFill.count&&this.faceFill.buffer&&(t.setPipeline(c(o)),t.setBindGroup(0,l),t.setVertexBuffer(0,this.faceFill.buffer),t.draw(this.faceFill.count)),this.faceOutline.count&&this.faceOutline.buffer&&(t.setPipeline(c(a)),t.setBindGroup(0,l),t.setVertexBuffer(0,this.faceOutline.buffer),t.draw(6,this.faceOutline.count)),d&&(t.setPipeline(a.tested),t.setBindGroup(0,l),t.setVertexBuffer(0,this.edges.buffer),t.draw(6,this.edges.count)),h&&(t.setPipeline(s.tested),t.setBindGroup(0,l),t.setVertexBuffer(0,this.handles.buffer),t.draw(6,this.handles.count)),this.gizmo.count&&this.gizmo.buffer)){const[f,m]=this.visibleGizmoMode==="translate"?[r.gizmoArrowPipeline,9]:this.visibleGizmoMode==="scale"?[r.gizmoScalePipeline,12]:[r.gizmoEdgePipeline,6];t.setPipeline(f),t.setBindGroup(0,l),t.setVertexBuffer(0,this.gizmo.buffer),t.draw(m,this.gizmo.count)}}destroy(){for(const t of[this.edges,this.handles,this.gizmo,this.faceFill,this.faceOutline,this.meshEdgeSegments])t.destroy()}}const Ep=`// WGSL for the viewport's construction grid — where the world's planes are.
//
// One fullscreen triangle, emitted at the far plane (z = 1) and depth-tested
// \`less-equal\` against the depth the SDF pass writes. A ray miss writes exactly
// 1.0, so the grid paints on the background and is rejected wherever any
// geometry, FEM surface or path-traced silhouette wrote a nearer depth. That is
// what "drawn behind the geometry so the part occludes it" means here, and it
// costs no sorting, no readback and no second pass over the scene.
//
// What the fragment does is raycast a world coordinate plane. The camera basis
// arrives as uniforms, each fragment reconstructs the same ray the preview
// shader would have used, intersects it with the plane and rules a square grid
// in *world* coordinates on it. The grid therefore recedes with perspective and
// slides under the part as you orbit, which is the whole reason it tells you
// anything about where things are.
//
// Which plane is not fixed. On z = 0 the grid is right in Top and Bottom and
// useless in Front, Back, Left and Right, where it is exactly edge-on and rules
// as one line — the four views someone reaches for to measure something square.
// \`plane_weights\` therefore scores all three world planes by how face-on they
// are, biased toward the floor because the floor is the model's ground and the
// rest of the viewport references it, and rules the winner. The two crossovers
// are a few degrees wide and dissolve, so the plane changes without a jump; the
// readout in \`components/viewer/Graticule.tsx\` names whichever one is up. The
// arithmetic is mirrored on the CPU in \`viewer/graticule.ts\` — one of them
// draws the grid, the other names it, and they must agree.
//
// Two things keep the horizon from turning into moiré. Coverage is measured in
// pixels through \`fwidth\`, so a line is one pixel wide however oblique the
// plane is; and the pattern fades out both with distance from the orbit target
// and as the cell size falls below a few pixels, so the far field dissolves
// instead of aliasing.
//
// It is deliberately faint — a minor line is about 1.35:1 against the paper,
// below the band structure is held to. It is a spatial cue, not structure, and
// it has to disappear the moment attention lands on the geometry over it.
//
// The colours arrive as uniforms from the token layer; there are no literals.

struct Graticule {
  // width, height (framebuffer px), line width (px), 1 when orthographic
  frame   : vec4<f32>,
  // camera position xyz, orthographic frame height in world units
  eye     : vec4<f32>,
  // camera right xyz, viewport aspect
  right   : vec4<f32>,
  // camera up xyz, grid spacing in world units
  up      : vec4<f32>,
  // camera forward xyz, the projection's field scale
  forward : vec4<f32>,
  // minor grid lines: rgb + alpha
  line    : vec4<f32>,
  // every nth line, drawn a step firmer: rgb + alpha
  major   : vec4<f32>,
  // the two in-plane axes: rgb + alpha
  axis    : vec4<f32>,
  // fade start, fade end (world radius), axis width multiple, lines per major
  fade    : vec4<f32>,
  // orbit centre xyz — the fade is measured from it, unused w
  centre  : vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Graticule;

/// How much the floor is preferred over a wall. See \`GRID_FLOOR_PREFERENCE\`.
const FLOOR_PREFERENCE: f32 = 2.0;

/// Half-width of a crossover, in score units. See \`GRID_PLANE_BAND\`.
const PLANE_BAND: f32 = 0.06;

@vertex
fn vs_graticule(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4<f32> {
  // One oversized triangle covering the clip rectangle, at the far plane.
  var corners = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
  );
  return vec4<f32>(corners[vertex_index], 1.0, 1.0);
}

/// Coverage of a line of half-width \`half_width\` at distance \`distance\` px.
fn hairline(distance: f32, half_width: f32) -> f32 {
  return clamp(half_width - distance + 0.5, 0.0, 1.0);
}

/// Straight "over" compositing, non-premultiplied, so the pass can blend once.
fn over(dst: vec4<f32>, src_rgb: vec3<f32>, src_a: f32) -> vec4<f32> {
  let alpha = src_a + dst.a * (1.0 - src_a);
  if (alpha <= 1e-6) {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }
  let rgb = (src_rgb * src_a + dst.rgb * dst.a * (1.0 - src_a)) / alpha;
  return vec4<f32>(rgb, alpha);
}

/// The two world axes a plane is spanned by, and the one it is normal to.
struct PlaneAxes {
  u : vec3<f32>,
  v : vec3<f32>,
  n : vec3<f32>,
};

/// 0 = XY (the floor), 1 = XZ, 2 = YZ — the order \`GRID_PLANES\` uses.
fn plane_axes(kind: i32) -> PlaneAxes {
  let x = vec3<f32>(1.0, 0.0, 0.0);
  let y = vec3<f32>(0.0, 1.0, 0.0);
  let z = vec3<f32>(0.0, 0.0, 1.0);
  var axes: PlaneAxes;
  axes.u = x;
  axes.v = y;
  axes.n = z;
  if (kind == 1) {
    axes.v = z;
    axes.n = y;
  } else if (kind == 2) {
    axes.u = y;
    axes.v = z;
    axes.n = x;
  }
  return axes;
}

/// How strongly each world plane is drawn, in \`plane_axes\` order.
///
/// A plane is worth ruling in proportion to how square it is to the camera,
/// |forward · n|; the floor carries \`FLOOR_PREFERENCE\` on top of that. The
/// smoothstep against the best rival is what makes the change a dissolve: it
/// is 1 for the leader and 0 for the rest except within \`PLANE_BAND\` of a
/// crossover, where two of them share the paint. Normalised, so the grid never
/// gets fainter or firmer for changing plane.
fn plane_weights(forward: vec3<f32>) -> vec3<f32> {
  let face = abs(forward);
  let score = vec3<f32>(face.z * FLOOR_PREFERENCE, face.y, face.x);
  let rival = vec3<f32>(
    max(score.y, score.z),
    max(score.x, score.z),
    max(score.x, score.y),
  );
  let raw = smoothstep(
    vec3<f32>(-PLANE_BAND),
    vec3<f32>(PLANE_BAND),
    score - rival,
  );
  let total = raw.x + raw.y + raw.z;
  if (total <= 1e-6) {
    return vec3<f32>(1.0, 0.0, 0.0);
  }
  return raw / total;
}

/// Rule one world plane into the accumulating paint.
///
/// Everything below is in the plane's own (u, v) coordinates, so the floor and
/// the two walls are the same code: the grid, the every-fifth major line, and
/// the two axes where this plane meets the other two.
fn rule_plane(
  dst: vec4<f32>,
  kind: i32,
  weight: f32,
  ray_origin: vec3<f32>,
  ray_dir: vec3<f32>,
) -> vec4<f32> {
  let axes = plane_axes(kind);
  let orthographic = g.frame.w > 0.5;
  let half_width = g.frame.z * 0.5;
  let spacing = max(g.up.w, 1e-6);

  // Intersect the plane. \`t\` is the distance along a unit ray, so under
  // perspective it doubles as the depth in front of the eye — and a hit behind
  // the eye is the sky, not the floor. Under orthographic there is no eye: the
  // origin is a station on the camera plane, the ray is one line through the
  // world, and the plane point on it is on screen whichever side of the
  // station it falls. Testing \`t > 0\` there is what cut the near third of the
  // ground off along a hard horizontal line.
  let denominator = dot(ray_dir, axes.n);
  let along = -dot(ray_origin, axes.n)
    / select(denominator, 1e-6, abs(denominator) < 1e-6);
  let facing = abs(denominator) >= 1e-6;
  let hit = select(0.0, 1.0, facing && (orthographic || along > 0.0));
  let ground = ray_origin + ray_dir * along;
  let plane = vec2<f32>(dot(ground, axes.u), dot(ground, axes.v));
  let centre = vec2<f32>(dot(g.centre.xyz, axes.u), dot(g.centre.xyz, axes.v));

  // Derivatives are taken unconditionally: WGSL requires uniform control flow
  // for them, and a quad that straddles the horizon simply reports an enormous
  // width, which the coverage below turns into nothing. That is the correct
  // answer there anyway.
  let cell = plane / spacing;
  let cell_width = fwidth(cell);
  let plane_width = fwidth(plane);

  // Distance to the nearest ruled line, converted from cells to pixels, so a
  // line is one pixel wide no matter how oblique the plane is.
  let to_line = abs(cell - round(cell));
  let line_px = to_line / max(cell_width, vec2<f32>(1e-8));
  let minor = hairline(min(line_px.x, line_px.y), half_width);

  // Every nth line again, a step firmer — the only hierarchy the plane has,
  // and what lets you count cells without reading each one.
  let major_cell = cell / max(g.fade.w, 1.0);
  let major_width = fwidth(major_cell);
  let major_to_line = abs(major_cell - round(major_cell));
  let major_px = major_to_line / max(major_width, vec2<f32>(1e-8));
  let major = hairline(min(major_px.x, major_px.y), half_width);

  // Fade outward from the orbit target, not from the eye: what you are looking
  // at is the part of the plane worth ruling, and everything past it dissolves
  // rather than aliasing into moiré at the horizon. The second fade catches the
  // same failure arriving from the other direction — a steep view close up,
  // where a cell falls below a couple of pixels.
  let radius = length(plane - centre);
  let distance_fade = 1.0 - smoothstep(g.fade.x, g.fade.y, radius);
  let density_fade = 1.0 - smoothstep(0.125, 0.5, max(cell_width.x, cell_width.y));
  let major_density = 1.0 - smoothstep(0.125, 0.5, max(major_width.x, major_width.y));
  let visibility = hit * distance_fade * weight;

  var paint = over(dst, g.line.rgb, minor * g.line.a * visibility * density_fade);
  paint = over(paint, g.major.rgb, major * g.major.a * visibility * major_density);

  // The two axes this plane carries: the lines where it meets the other two
  // coordinate planes, so the floor shows X and Y and a wall shows the vertical
  // one and its own horizontal. A step above a major line and not subject to
  // the density fade — they are two marks, not a pattern, so they cannot moiré,
  // and they are what says which way the world is pointing once the grid itself
  // has faded out.
  let axis_px = abs(plane) / max(plane_width, vec2<f32>(1e-8));
  let axis_half = half_width * g.fade.z;
  let axis = max(hairline(axis_px.y, axis_half), hairline(axis_px.x, axis_half));
  return over(paint, g.axis.rgb, axis * g.axis.a * visibility);
}

@fragment
fn fs_graticule(@builtin(position) frag : vec4<f32>) -> @location(0) vec4<f32> {
  let size = g.frame.xy;
  let orthographic = g.frame.w > 0.5;
  let aspect = g.right.w;

  // The same field coordinates the projection in \`viewer/math.ts\` uses:
  // px = (u / aspect + 0.5) * width, py = (0.5 - v) * height.
  let u = (frag.x / size.x - 0.5) * aspect;
  let v = 0.5 - frag.y / size.y;
  let lateral = g.right.xyz * u + g.up.xyz * v;

  // Perspective rays fan out from the eye; orthographic rays are parallel and
  // the fan becomes an offset of the origin. One \`select\` rather than a branch,
  // because the derivatives below need uniform control flow.
  let ray_origin = select(g.eye.xyz, g.eye.xyz + lateral * g.eye.w, orthographic);
  let ray_dir = select(
    normalize(g.forward.xyz + lateral * g.forward.w),
    g.forward.xyz,
    orthographic,
  );

  // All three planes are ruled unconditionally and weighted, rather than
  // branched on: the weights are a function of a uniform, so two of them are
  // zero everywhere except in a crossover, and unrolling keeps every \`fwidth\`
  // below in flat, uniform control flow.
  let weights = plane_weights(g.forward.xyz);
  var paint = paint_zero();
  paint = rule_plane(paint, 0, weights.x, ray_origin, ray_dir);
  paint = rule_plane(paint, 1, weights.y, ray_origin, ray_dir);
  paint = rule_plane(paint, 2, weights.z, ray_origin, ray_dir);
  return paint;
}

/// The empty accumulator. A function so the compositing chain reads as one
/// expression from nothing to the final colour.
fn paint_zero() -> vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
`,Ip=`// WGSL for the construction-tree overlay.
// 
// Edges and vertex handles are drawn as screen-space expanded quads so their
// width is in pixels rather than world units, and both are depth-tested against
// the depth the SDF pass writes — an edge behind the solid is correctly hidden.
// 
// Sketch edges usually lie exactly on the extruded solid's side faces, so every
// vertex is nudged a fixed fraction of the way toward the camera before
// projection. That is a scale-free bias: it resolves the coplanar z-fight at
// any zoom without a depth-bias constant tuned per scene.

struct Overlay {
  view_proj  : mat4x4<f32>,
  camera_pos : vec4<f32>,
  // width, height, line width (px), handle radius (px)
  viewport   : vec4<f32>,
  // depth nudge fraction, casing (px per side; 0 = none), unused, unused
  params     : vec4<f32>,
  // camera forward axis; w = 1 under an orthographic projection
  view_dir   : vec4<f32>,
};
@group(0) @binding(0) var<uniform> o: Overlay;

// How opaque a stroke's sheet casing is: enough to part a stroke from a dark
// facet, short of a solid white band.
const CASING_ALPHA = 0.75;
// Hidden skeleton: a 5 px dash in an 8 px period, at 45 % strength.
const HIDDEN_DASH_PERIOD = 8.0;
const HIDDEN_DASH_ON = 0.625;
const HIDDEN_ALPHA = 0.45;

fn half_viewport() -> vec2<f32> {
  return max(o.viewport.xy * 0.5, vec2<f32>(1.0));
}

// Pull a point slightly toward the camera so overlay geometry wins the depth
// test against surfaces it is coplanar with.
fn nudge(p: vec3<f32>) -> vec3<f32> {
  return p + (o.camera_pos.xyz - p) * o.params.x;
}

fn project(p: vec3<f32>) -> vec4<f32> {
  var clip = o.view_proj * vec4<f32>(nudge(p), 1.0);
  // Keep points behind the eye from wrapping to the far side of the screen.
  clip.w = max(clip.w, 1e-4);
  return clip;
}

struct EdgeVertex {
  @builtin(position) position : vec4<f32>,
  @location(0) side           : f32,
  @location(1) color          : vec4<f32>,
  // Screen pixels from the segment's start, for the hidden pass's dashes.
  // Linear, not perspective-correct: a dash is a screen-space pattern.
  @location(2) @interpolate(linear) along : f32,
};

// Half the drawn width of a line in pixels: the stroke, plus its casing.
fn edge_half_width() -> f32 {
  return o.viewport.z * 0.5 + max(o.params.y, 0.0);
}

@vertex
fn vs_edge(
  @builtin(vertex_index) vertex_index : u32,
  @location(0) start : vec3<f32>,
  @location(1) end   : vec3<f32>,
  @location(2) color : vec4<f32>,
) -> EdgeVertex {
  // Two triangles spanning the segment: x picks the endpoint, y the side.
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
  );
  let corner = corners[vertex_index];

  let clip_start = project(start);
  let clip_end = project(end);
  let half_size = half_viewport();
  let screen_start = clip_start.xy / clip_start.w * half_size;
  let screen_end = clip_end.xy / clip_end.w * half_size;

  var direction = screen_end - screen_start;
  let span = length(direction);
  direction = select(vec2<f32>(1.0, 0.0), direction / span, span > 1e-6);
  let normal = vec2<f32>(-direction.y, direction.x);

  let base = select(clip_start, clip_end, corner.x > 0.5);
  let offset = normal * edge_half_width() * corner.y / half_size;

  var out: EdgeVertex;
  out.position = vec4<f32>(base.xy + offset * base.w, base.z, base.w);
  out.side = corner.y;
  out.color = color;
  out.along = corner.x * span;
  return out;
}

@fragment
fn fs_edge(input: EdgeVertex) -> @location(0) vec4<f32> {
  let side = abs(input.side);
  if (o.params.y <= 0.0) {
    // No casing (mesh edges): feather the outer pixel so lines do not alias
    // into stair steps.
    let coverage = smoothstep(1.0, 0.55, side);
    return vec4<f32>(input.color.rgb, input.color.a * coverage);
  }
  // The stroke, cased in sheet. A hue alone separates the skeleton from a
  // grey part only where the two differ in value as well; the casing is a
  // pale keyline either side, so the stroke has a light edge over a dark
  // facet and the same light edge over a light one.
  let half_width = edge_half_width();
  let feather = 0.55 / half_width;
  let core = (o.viewport.z * 0.5) / half_width;
  let outer = smoothstep(1.0, 1.0 - feather, side);
  let stroke = smoothstep(core + feather * 0.5, core - feather * 0.5, side);
  let shade = mix(SHEET, input.color.rgb, stroke);
  return vec4<f32>(shade, mix(CASING_ALPHA, input.color.a, stroke) * outer);
}

// The skeleton where the part hides it (x-ray only): the stroke without its
// casing, dashed, and at a fraction of its strength, so what is behind the
// solid never carries the weight of what is in front of it.
@fragment
fn fs_edge_hidden(input: EdgeVertex) -> @location(0) vec4<f32> {
  let half_width = edge_half_width();
  let core = (o.viewport.z * 0.5) / half_width;
  let coverage = smoothstep(core, core - 0.55 / half_width, abs(input.side));
  let dash = step(fract(input.along / HIDDEN_DASH_PERIOD), HIDDEN_DASH_ON);
  return vec4<f32>(input.color.rgb, input.color.a * coverage * dash * HIDDEN_ALPHA);
}

// Translation axes get their own screen-space shape instead of being assembled
// from wireframe segments. The result is a compact filled arrow whose weight is
// stable at every zoom level.
struct GizmoVertex {
  @builtin(position) position : vec4<f32>,
  @location(0) color          : vec4<f32>,
  @location(1) side           : f32,
  @location(2) barycentric    : vec3<f32>,
  @location(3) head           : f32,
};

@vertex
fn vs_gizmo_arrow(
  @builtin(vertex_index) vertex_index : u32,
  @location(0) start    : vec3<f32>,
  @location(1) end      : vec3<f32>,
  @location(2) color    : vec4<f32>,
  @location(3) emphasis : f32,
) -> GizmoVertex {
  let clip_start = project(start);
  let clip_end = project(end);
  let half_size = half_viewport();
  let screen_start = clip_start.xy / clip_start.w * half_size;
  let screen_end = clip_end.xy / clip_end.w * half_size;

  var direction = screen_end - screen_start;
  let span = max(length(direction), 1e-4);
  direction = select(vec2<f32>(1.0, 0.0), direction / span, span > 1e-4);
  let normal = vec2<f32>(-direction.y, direction.x);

  let head_length = min(14.0 + emphasis * 2.0, span * 0.42);
  let head_width = 6.5 + emphasis * 1.5;
  let shaft_width = 2.35 + emphasis * 0.65;
  let start_gap = min(5.0, span * 0.12);
  let head_base = screen_end - direction * head_length;
  let shaft_start = screen_start + direction * start_gap;
  // A tiny overlap prevents a hairline seam between the shaft and head.
  let shaft_end = head_base + direction;

  var point: vec2<f32>;
  var along = 0.0;
  var side = 0.0;
  var barycentric = vec3<f32>(-1.0);
  var head = 0.0;

  if (vertex_index < 6u) {
    var corners = array<vec2<f32>, 6>(
      vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
      vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let corner = corners[vertex_index];
    point = mix(shaft_start, shaft_end, corner.x) + normal * shaft_width * corner.y;
    along = (start_gap + corner.x * max(span - start_gap - head_length + 1.0, 0.0)) / span;
    side = corner.y;
  } else {
    let head_vertex = vertex_index - 6u;
    var points = array<vec2<f32>, 3>(
      screen_end,
      head_base + normal * head_width,
      head_base - normal * head_width,
    );
    var barycentrics = array<vec3<f32>, 3>(
      vec3<f32>(1.0, 0.0, 0.0),
      vec3<f32>(0.0, 1.0, 0.0),
      vec3<f32>(0.0, 0.0, 1.0),
    );
    point = points[head_vertex];
    along = select(1.0 - head_length / span, 1.0, head_vertex == 0u);
    barycentric = barycentrics[head_vertex];
    head = 1.0;
  }

  let clip = mix(clip_start, clip_end, clamp(along, 0.0, 1.0));
  var out: GizmoVertex;
  out.position = vec4<f32>(point / half_size * clip.w, clip.z, clip.w);
  out.color = color;
  out.side = side;
  out.barycentric = barycentric;
  out.head = head;
  return out;
}

@vertex
fn vs_gizmo_scale(
  @builtin(vertex_index) vertex_index : u32,
  @location(0) start    : vec3<f32>,
  @location(1) end      : vec3<f32>,
  @location(2) color    : vec4<f32>,
  @location(3) emphasis : f32,
) -> GizmoVertex {
  let clip_start = project(start);
  let clip_end = project(end);
  let half_size = half_viewport();
  let screen_start = clip_start.xy / clip_start.w * half_size;
  let screen_end = clip_end.xy / clip_end.w * half_size;

  var direction = screen_end - screen_start;
  let span = max(length(direction), 1e-4);
  direction = select(vec2<f32>(1.0, 0.0), direction / span, span > 1e-4);
  let normal = vec2<f32>(-direction.y, direction.x);
  let block_radius = 5.2 + emphasis;
  let shaft_width = 2.2 + emphasis * 0.65;
  let start_gap = min(5.0, span * 0.12);
  let shaft_start = screen_start + direction * start_gap;
  let shaft_end = screen_end - direction * (block_radius - 0.8);

  var point: vec2<f32>;
  var along = 0.0;
  var side = 0.0;
  var local = vec2<f32>(-1.0);
  var head = 0.0;

  if (vertex_index < 6u) {
    var corners = array<vec2<f32>, 6>(
      vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
      vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let corner = corners[vertex_index];
    point = mix(shaft_start, shaft_end, corner.x) + normal * shaft_width * corner.y;
    along = (start_gap + corner.x * max(span - start_gap - block_radius + 0.8, 0.0)) / span;
    side = corner.y;
  } else {
    let block_vertex = vertex_index - 6u;
    var corners = array<vec2<f32>, 6>(
      vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
      vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    local = corners[block_vertex];
    point = screen_end + (direction * local.x + normal * local.y) * block_radius;
    along = 1.0;
    head = 2.0;
  }

  let clip = mix(clip_start, clip_end, clamp(along, 0.0, 1.0));
  var out: GizmoVertex;
  out.position = vec4<f32>(point / half_size * clip.w, clip.z, clip.w);
  out.color = color;
  out.side = side;
  out.barycentric = vec3<f32>(local, 0.0);
  out.head = head;
  return out;
}

@fragment
fn fs_gizmo_arrow(input: GizmoVertex) -> @location(0) vec4<f32> {
  var border = smoothstep(0.62, 0.86, abs(input.side));
  var coverage = smoothstep(1.0, 0.82, abs(input.side));
  if (input.head > 1.5) {
    let edge = max(abs(input.barycentric.x), abs(input.barycentric.y));
    border = smoothstep(0.64, 0.86, edge);
    coverage = smoothstep(1.0, 0.88, edge);
  } else if (input.head > 0.5) {
    let nearest_edge = min(input.barycentric.x, min(input.barycentric.y, input.barycentric.z));
    border = 1.0 - smoothstep(0.055, 0.14, nearest_edge);
    coverage = 1.0;
  }
  // The arrow's keyline is ink (the \`INK\` token), the same mark every other
  // overlay is drawn in; only the body carries the axis hue.
  let shaded = mix(input.color.rgb, INK, border * 0.82);
  return vec4<f32>(shaded, input.color.a * coverage);
}

struct HandleVertex {
  @builtin(position) position : vec4<f32>,
  @location(0) local          : vec2<f32>,
  @location(1) color          : vec4<f32>,
  @location(2) emphasis       : f32,
  // HANDLE_SHAPE in overlayGeometry.ts: 0 = free (hollow circle), 1 = held
  // (filled square), 2 = selected (filled circle). Free versus held is told
  // apart by shape, never by hue (V1).
  @location(3) shape          : f32,
};

@vertex
fn vs_handle(
  @builtin(vertex_index) vertex_index : u32,
  @location(0) center   : vec3<f32>,
  @location(1) color    : vec4<f32>,
  @location(2) emphasis : f32,
  @location(3) shape    : f32,
) -> HandleVertex {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
  );
  let corner = corners[vertex_index];
  let clip = project(center);
  let radius = o.viewport.w * (1.0 + emphasis * 0.35);
  let offset = corner * radius / half_viewport();

  var out: HandleVertex;
  out.position = vec4<f32>(clip.xy + offset * clip.w, clip.z, clip.w);
  out.local = corner;
  out.color = color;
  out.emphasis = emphasis;
  out.shape = shape;
  return out;
}

// One handle's colour and coverage at a point of its quad. Every shape has
// a sheet casing round it, for the same reason a stroke does: a filled
// square in the construction hue over a facet of the same value would be a
// smudge, and a pale rim makes it a mark over any shade.
fn handle_shade(input: HandleVertex) -> vec4<f32> {
  if (input.shape > 0.5 && input.shape < 1.5) {
    // Held: a filled square, a little inside the circle's radius so the two
    // shapes carry the same visual weight.
    let d = max(abs(input.local.x), abs(input.local.y)) / 0.86;
    if (d > 1.0) {
      discard;
    }
    let edge = smoothstep(1.0, 0.86, d);
    let body = smoothstep(0.80, 0.72, d);
    let shade = mix(SHEET, input.color.rgb, body);
    return vec4<f32>(shade, mix(CASING_ALPHA, input.color.a, body) * edge);
  }
  let radius = length(input.local);
  if (radius > 1.0) {
    discard;
  }
  let edge = smoothstep(1.0, 0.82, radius);
  if (input.shape > 1.5) {
    // Selected: a filled disc in the handle's colour (ink), cased.
    let body = smoothstep(0.80, 0.70, radius);
    let shade = mix(SHEET, input.color.rgb, body);
    return vec4<f32>(shade, mix(CASING_ALPHA, input.color.a, body) * edge);
  }
  // Free: a ring in the handle's colour around an opaque sheet middle, so the
  // hollow reads as hollow over a dark solid as well as over paper, inside the
  // same sheet casing as the filled shapes. The band is wide enough (~0.3 of
  // the radius) to survive at handle size.
  let ring = smoothstep(0.46, 0.56, radius) * smoothstep(0.86, 0.78, radius);
  let shade = mix(SHEET, input.color.rgb, ring);
  return vec4<f32>(shade, input.color.a * edge);
}

@fragment
fn fs_handle(input: HandleVertex) -> @location(0) vec4<f32> {
  return handle_shade(input);
}

// A handle the part hides (x-ray only): the same shape at a fraction of its
// strength, like the hidden skeleton it sits on.
@fragment
fn fs_handle_hidden(input: HandleVertex) -> @location(0) vec4<f32> {
  let shade = handle_shade(input);
  return vec4<f32>(shade.rgb, shade.a * HIDDEN_ALPHA);
}

// ── face highlight ──────────────────────────────────────────────────────────
//
// The one overlay drawn as an actual surface rather than as an expanded quad:
// the polygon of the face under the pointer, triangulated on the CPU, filled
// with a faint achromatic wash. It carries the same camera nudge as the edges,
// so it wins the depth test against the face it is painted on, and it does not
// write depth — the hairline outline drawn straight after it has to land on
// top, not z-fight with the wash it belongs to.

struct FaceVertex {
  @builtin(position) position : vec4<f32>,
  @location(0)       color    : vec4<f32>,
};

@vertex
fn vs_face(@location(0) point: vec3<f32>, @location(1) color: vec4<f32>) -> FaceVertex {
  var out: FaceVertex;
  out.position = project(point);
  out.color = color;
  return out;
}

@fragment
fn fs_face(input: FaceVertex) -> @location(0) vec4<f32> {
  return input.color;
}

// ── B-rep inspector faces ───────────────────────────────────────────────────
//
// The derived B-rep's own tessellation, drawn *instead of* the raymarched
// solid while the inspector is on: what the extraction found, facet for
// facet, one categorical colour per face. Opaque and depth-writing, with no
// camera nudge — it is the surface, not something drawn on one — so the
// B-rep's edges and vertices that follow land on it like overlays on the
// solid. Shaded with the surface's own normal at every triangle corner when
// the private tier supplies one, so a smooth blend shades smooth and a
// crease stays sharp (each face carries its own normal there); without one,
// flat from screen-space derivatives, the simulation surface's trick.

struct BrepVertex {
  @builtin(position) position : vec4<f32>,
  @location(0)       color    : vec4<f32>,
  @location(1)       world    : vec3<f32>,
  @location(2)       normal   : vec3<f32>,
};

@vertex
fn vs_brep(
  @location(0) point: vec3<f32>,
  @location(1) color: vec4<f32>,
  @location(2) normal: vec3<f32>,
) -> BrepVertex {
  var out: BrepVertex;
  var clip = o.view_proj * vec4<f32>(point, 1.0);
  clip.w = max(clip.w, 1e-4);
  out.position = clip;
  out.color = color;
  out.world = point;
  out.normal = normal;
  return out;
}

@fragment
fn fs_brep(input: BrepVertex) -> @location(0) vec4<f32> {
  let facet = normalize(cross(dpdx(input.world), dpdy(input.world)));
  // A zero normal means none was supplied: fall back to the facet's.
  let normal = select(facet, normalize(input.normal), dot(input.normal, input.normal) > 0.25);
  // Toward the eye: the eye point under perspective, and under an
  // orthographic projection the one direction every ray shares — from the
  // eye point a flat plane shaded as a radial glow.
  let eye = select(
    normalize(o.camera_pos.xyz - input.world),
    -normalize(o.view_dir.xyz),
    o.view_dir.w > 0.5,
  );
  // Facing ratio, sign-free: a triangle wound either way shades the same,
  // so a face the tessellation turned inside out is still readable.
  let facing = abs(dot(normal, eye));
  let shade = 0.62 + 0.38 * facing;
  return vec4<f32>(input.color.rgb * shade, 1.0);
}
`,Ap=`// WGSL for the FEM simulation surface.
//
// Draws the boundary of the solved hex mesh as an indexed triangle list with
// one scalar per vertex (temperature or von Mises stress). The fragment stage
// maps the scalar through a baked viridis ramp, shades with a screen-space
// derivative normal (flat facets, no normal buffer needed), and supports a
// ParaView-style clip plane: fragments beyond the plane are discarded so the
// interior of the field is visible along any axis.

struct SimUniforms {
  view_proj : mat4x4<f32>,
  // Clip plane: fragments with dot(world, xyz) > w are discarded.
  clip      : vec4<f32>,
  // min scalar, 1 / (max - min), outline offset, clip enabled flag.
  // z is 0 for every pass but the hover outline, where it is the size of one
  // framebuffer pixel in NDC (see vs_sim_edge).
  params    : vec4<f32>,
  // x: ramp selector (0 = field/viridis, 1 = quality/magma).
  // yzw: camera position in world space. The facet normal comes from
  // screen-space derivatives, whose sign depends on the derivative order, so
  // the eye vector is needed both to orient it and to find the silhouette.
  extra     : vec4<f32>,
};
@group(0) @binding(0) var<uniform> s: SimUniforms;

struct SimVertex {
  @builtin(position) position : vec4<f32>,
  @location(0) world          : vec3<f32>,
  @location(1) scalar         : f32,
  // Per-vertex highlight tint (BC previews): rgb hue + blend strength.
  @location(2) overlay        : vec4<f32>,
};

@vertex
fn vs_sim(
  @location(0) position : vec3<f32>,
  @location(1) scalar   : f32,
  @location(2) overlay  : vec4<f32>,
) -> SimVertex {
  var out: SimVertex;
  out.position = s.view_proj * vec4<f32>(position, 1.0);
  out.world = position;
  out.scalar = scalar;
  out.overlay = overlay;
  return out;
}

// Polynomial fits of the matplotlib colormaps (degree 6, per channel). The
// coefficients are the VIRIDIS_C* / MAGMA_C* constants \`wgslTokens()\` prepends
// from tokens.ts — the same tables the legends in src/simColors.ts evaluate.
// viridis is the FIELD ramp, magma the QUALITY ramp.
fn viridis(t: f32) -> vec3<f32> {
  return VIRIDIS_C0 + t * (VIRIDIS_C1 + t * (VIRIDIS_C2 + t * (VIRIDIS_C3
    + t * (VIRIDIS_C4 + t * (VIRIDIS_C5 + t * VIRIDIS_C6)))));
}

fn magma(t: f32) -> vec3<f32> {
  return MAGMA_C0 + t * (MAGMA_C1 + t * (MAGMA_C2 + t * (MAGMA_C3
    + t * (MAGMA_C4 + t * (MAGMA_C5 + t * MAGMA_C6)))));
}

@fragment
fn fs_sim(input: SimVertex) -> @location(0) vec4<f32> {
  if (s.params.w > 0.5 && dot(input.world, s.clip.xyz) > s.clip.w) {
    discard;
  }
  let t = clamp((input.scalar - s.params.x) * s.params.y, 0.0, 1.0);
  var color: vec3<f32>;
  if (s.extra.x > 0.5) {
    color = clamp(magma(t), vec3<f32>(0.0), vec3<f32>(1.0));
  } else {
    color = clamp(viridis(t), vec3<f32>(0.0), vec3<f32>(1.0));
  }
  // BC-preview tint, blended over the field per vertex. That one is data (a
  // BC hue or the proposal), so it may colour the surface; hover may not.
  // Over a solved field, hover and selection are drawn, not tinted (V2): the
  // hovered face group gets an ink outline from the edge pipeline below
  // instead of a warm wash, so the field value under the pointer is the one
  // the ramp says it is.
  color = mix(color, input.overlay.rgb, clamp(input.overlay.a, 0.0, 1.0));
  // Flat facet shading from screen-space derivatives; clipped cross sections
  // keep their facet normal too, which reads like a real cut.
  //
  // On the paper viewport the field can no longer be read by its own
  // brightness: viridis(1) is 1.02:1 against \`#e6e6e9\` and magma(1) is
  // 1.15:1, so a hot region and the ground are the same luminance. Two
  // changes carry the form instead.
  //
  // 1. A signed Lambert term rather than \`abs(dot(...))\`. \`abs\` was a
  //    dark-ground trick — it kept every facet emitting — but it folds the
  //    normal sphere in half, so a facet turned away from the key light was
  //    drawn as bright as one turned into it. With the eye vector in
  //    \`extra.yzw\` the derivative normal can be oriented, so the term can be
  //    signed and faces pointing away can go properly dark. The key light
  //    also moved off the diagonal: at (0.55, 0.8, 0.35) the three visible
  //    faces of an axis-aligned part all sat within 0.15 of each other, and
  //    on paper facet separation is most of what draws the form.
  // 2. A silhouette contour. The outermost sliver of the surface — where the
  //    facet turns away from the eye — is darkened hard. That is the only
  //    thing separating a yellow field from a white ground, and it costs no
  //    colour fidelity anywhere but the rim.
  let eye = normalize(s.extra.yzw - input.world);
  var normal = normalize(cross(dpdx(input.world), dpdy(input.world)));
  normal = normal * sign(dot(normal, eye) + 1e-6);
  let light = normalize(vec3<f32>(0.30, 0.86, 0.42));
  let shade = 0.42 + 0.58 * max(dot(normal, light), 0.0);
  let contour = mix(0.32, 1.0, smoothstep(0.0, 0.2, abs(dot(normal, eye))));
  return vec4<f32>(color * shade * contour, 1.0);
}

// ── element edges: the mesh's boundary-face edge lines ───────────────────
// Same vertex buffer as the surface (position at offset 0), drawn as a
// line list over its own index buffer. A small clip-space nudge keeps the
// hairlines in front of their own faces.

struct SimEdgeVertex {
  @builtin(position) position : vec4<f32>,
  @location(0) world          : vec3<f32>,
  @location(1) scalar         : f32,
};

@vertex
fn vs_sim_edge(
  @builtin(instance_index) instance : u32,
  @location(0) position : vec3<f32>,
  @location(1) scalar   : f32,
) -> SimEdgeVertex {
  var out: SimEdgeVertex;
  out.position = s.view_proj * vec4<f32>(position, 1.0);
  // A larger nudge than the faces get: hairlines that z-fight with their
  // own surface read as dashed, which looks like missing elements.
  out.position.z = out.position.z - 0.004 * out.position.w;
  // The hover outline is the same line list drawn three times, each copy
  // shifted one pixel (params.z, in NDC): WebGPU lines are always one pixel
  // wide, and a one-pixel outline is indistinguishable from an element edge.
  // Instance 0 is unshifted, so the element-edge pass (one instance) is not.
  var shift = array<vec2<f32>, 3>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
  out.position = vec4<f32>(
    out.position.xy + shift[min(instance, 2u)] * s.params.z * out.position.w,
    out.position.zw,
  );
  out.world = position;
  out.scalar = scalar;
  return out;
}

@fragment
fn fs_sim_edge(input: SimEdgeVertex) -> @location(0) vec4<f32> {
  if (s.params.w > 0.5 && dot(input.world, s.clip.xyz) > s.clip.w) {
    discard;
  }
  // A single fixed hairline colour cannot stay legible across a whole ramp:
  // charcoal vanishes in viridis' dark end, white vanishes in its bright
  // end. Evaluate the same ramp the surface uses and pick whichever of the
  // two edge tones contrasts with it.
  // The hover outline (params.z > 0) runs through here too: the boundary of
  // the hovered face group, three pixels wide, in the same achromatic pair —
  // ink over the bright end of the ramp, sheet over the dark end, where ink
  // would vanish into viridis' near-black purple. It is the edge of the
  // region, not a recolouring of it; the field inside keeps its ramp value.
  let t = clamp((input.scalar - s.params.x) * s.params.y, 0.0, 1.0);
  var under = viridis(t);
  if (s.extra.x > 0.5) {
    under = magma(t);
  }
  let luminance = dot(clamp(under, vec3<f32>(0.0), vec3<f32>(1.0)),
                      vec3<f32>(0.2126, 0.7152, 0.0722));
  // ELEMENT_EDGE_COLOR / ELEMENT_EDGE_COLOR_LIGHT from tokens.ts: ink over
  // the bright end of the ramp, the sheet over the dark end. Both achromatic.
  if (luminance > 0.32) {
    return vec4<f32>(EDGE_DARK, 1.0);
  }
  return vec4<f32>(EDGE_LIGHT, 1.0);
}
`,Mp=`// The interaction preview, stretched back over the framebuffer.
//
// While a gesture is in flight a slow scene is raymarched into a smaller
// target (\`Renderer.gestureScale\`); this pass covers the framebuffer with it.
// Colour is filtered. Depth is not — a depth texture cannot be, and an
// averaged depth would be a surface nobody drew — so it is the nearest
// texel's, written back through \`frag_depth\` so the ground grid and the
// construction overlays that follow test against the solid as they always do.

@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var scene_sampler: sampler;

struct UpscaleVertex {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_upscale(@builtin(vertex_index) index: u32) -> UpscaleVertex {
  // One triangle over the whole viewport; uv runs 0..1 across it, top-left
  // first, the way texture coordinates do.
  let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
  var out: UpscaleVertex;
  out.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
  out.uv = uv;
  return out;
}

struct UpscaleFragment {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_upscale(in: UpscaleVertex) -> UpscaleFragment {
  let size = textureDimensions(scene_depth);
  let texel = min(vec2<u32>(in.uv * vec2<f32>(size)), size - vec2<u32>(1u));
  var out: UpscaleFragment;
  out.color = textureSampleLevel(scene_color, scene_sampler, in.uv, 0.0);
  out.depth = textureLoad(scene_depth, texel, 0);
  return out;
}
`;function Cr(e){return`${sl()}
${e}`}const He="depth32float";function si(e,t,n){const i=e.createBindGroupLayout({label:t,entries:n.map(r=>({binding:r,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}))});return{bindGroupLayout:i,pipelineLayout:e.createPipelineLayout({bindGroupLayouts:[i]})}}function Cp(e,t,n,i){const r=e.createShaderModule({code:Cr(Ip),label:"Overlay WGSL"}),{bindGroupLayout:a,pipelineLayout:s}=si(e,"Overlay bindings",[0]),o={color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}},l={format:He,depthWriteEnabled:!0,depthCompare:"less-equal"},c={format:He,depthWriteEnabled:!1,depthCompare:"always"},d={format:He,depthWriteEnabled:!1,depthCompare:"greater"},h=(_,$,E)=>({tested:e.createRenderPipeline({..._,depthStencil:$}),seen:e.createRenderPipeline({..._,label:`${_.label} (through the solid)`,depthStencil:c}),hidden:E&&_.fragment?e.createRenderPipeline({..._,label:`${_.label} (hidden by the solid)`,fragment:{..._.fragment,entryPoint:E},depthStencil:d}):void 0}),u=h({label:"Overlay edges",layout:s,vertex:{module:r,entryPoint:"vs_edge",buffers:[{arrayStride:an,stepMode:"instance",attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"},{shaderLocation:2,offset:24,format:"float32x4"}]}]},fragment:{module:r,entryPoint:"fs_edge",targets:[{format:t,blend:o}]},primitive:{topology:"triangle-list"}},l,"fs_edge_hidden"),p=h({label:"Overlay face highlight",layout:s,vertex:{module:r,entryPoint:"vs_face",buffers:[{arrayStride:Ks,stepMode:"vertex",attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x4"}]}]},fragment:{module:r,entryPoint:"fs_face",targets:[{format:t,blend:o}]},primitive:{topology:"triangle-list"}},{format:He,depthWriteEnabled:!1,depthCompare:"less-equal"}),f=e.createRenderPipeline({label:"B-rep inspector faces",layout:s,vertex:{module:r,entryPoint:"vs_brep",buffers:[{arrayStride:Zs,stepMode:"vertex",attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x4"},{shaderLocation:2,offset:28,format:"float32x3"}]}]},fragment:{module:r,entryPoint:"fs_brep",targets:[{format:t}]},primitive:{topology:"triangle-list"},depthStencil:l}),m=h({label:"Overlay handles",layout:s,vertex:{module:r,entryPoint:"vs_handle",buffers:[{arrayStride:Mr,stepMode:"instance",attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x4"},{shaderLocation:2,offset:28,format:"float32"},{shaderLocation:3,offset:32,format:"float32"}]}]},fragment:{module:r,entryPoint:"fs_handle",targets:[{format:t,blend:o}]},primitive:{topology:"triangle-list"}},l,"fs_handle_hidden"),w=e.createRenderPipeline({label:"Always-visible rotation gizmo",layout:s,vertex:{module:r,entryPoint:"vs_edge",buffers:[{arrayStride:Fn,stepMode:"instance",attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"},{shaderLocation:2,offset:24,format:"float32x4"}]}]},fragment:{module:r,entryPoint:"fs_edge",targets:[{format:t,blend:o}]},primitive:{topology:"triangle-list"},depthStencil:c}),x=e.createRenderPipeline({label:"Always-visible translation gizmo",layout:s,vertex:{module:r,entryPoint:"vs_gizmo_arrow",buffers:[{arrayStride:Fn,stepMode:"instance",attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"},{shaderLocation:2,offset:24,format:"float32x4"},{shaderLocation:3,offset:40,format:"float32"}]}]},fragment:{module:r,entryPoint:"fs_gizmo_arrow",targets:[{format:t,blend:o}]},primitive:{topology:"triangle-list"},depthStencil:c}),y=e.createRenderPipeline({label:"Always-visible scale gizmo",layout:s,vertex:{module:r,entryPoint:"vs_gizmo_scale",buffers:[{arrayStride:Fn,stepMode:"instance",attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"},{shaderLocation:2,offset:24,format:"float32x4"},{shaderLocation:3,offset:40,format:"float32"}]}]},fragment:{module:r,entryPoint:"fs_gizmo_arrow",targets:[{format:t,blend:o}]},primitive:{topology:"triangle-list"},depthStencil:c});return{edgePipeline:u,facePipeline:p,handlePipeline:m,brepFacePipeline:f,gizmoEdgePipeline:w,gizmoArrowPipeline:x,gizmoScalePipeline:y,overlayBindGroup:e.createBindGroup({layout:a,entries:[{binding:0,resource:{buffer:n}}]}),meshOverlayBindGroup:e.createBindGroup({layout:a,entries:[{binding:0,resource:{buffer:i}}]})}}const Tp=160;function Op(e,t){const n=e.createShaderModule({code:Mp,label:"Upscale WGSL"});return e.createRenderPipeline({label:"Interaction preview upscale",layout:"auto",vertex:{module:n,entryPoint:"vs_upscale"},fragment:{module:n,entryPoint:"fs_upscale",targets:[{format:t}]},primitive:{topology:"triangle-list"},depthStencil:{format:He,depthWriteEnabled:!0,depthCompare:"always"}})}function Lp(e,t,n){const i=e.createShaderModule({code:Cr(Ep),label:"Graticule WGSL"}),{bindGroupLayout:r,pipelineLayout:a}=si(e,"Graticule bindings",[0]);return{graticulePipeline:e.createRenderPipeline({label:"Viewport ground grid",layout:a,vertex:{module:i,entryPoint:"vs_graticule"},fragment:{module:i,entryPoint:"fs_graticule",targets:[{format:t,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}}]},primitive:{topology:"triangle-list"},depthStencil:{format:He,depthWriteEnabled:!1,depthCompare:"less-equal"}}),graticuleBindGroup:e.createBindGroup({layout:r,entries:[{binding:0,resource:{buffer:n}}]})}}function Dp(e,t,n){const i=e.createShaderModule({code:Cr(Ap),label:"Simulation WGSL"}),{bindGroupLayout:r,pipelineLayout:a}=si(e,"Simulation bindings",[0]),s=e.createRenderPipeline({label:"Simulation surface",layout:a,vertex:{module:i,entryPoint:"vs_sim",buffers:[{arrayStride:32,stepMode:"vertex",attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32"},{shaderLocation:2,offset:16,format:"float32x4"}]}]},fragment:{module:i,entryPoint:"fs_sim",targets:[{format:t}]},primitive:{topology:"triangle-list"},depthStencil:{format:He,depthWriteEnabled:!0,depthCompare:"less-equal"}}),o=e.createRenderPipeline({label:"Simulation element edges",layout:a,vertex:{module:i,entryPoint:"vs_sim_edge",buffers:[{arrayStride:32,stepMode:"vertex",attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32"}]}]},fragment:{module:i,entryPoint:"fs_sim_edge",targets:[{format:t}]},primitive:{topology:"line-list"},depthStencil:{format:He,depthWriteEnabled:!1,depthCompare:"less-equal"}}),l=n(112),c=n(112);return{simPipeline:s,simEdgePipeline:o,simUniformBuffer:l,simHighlightUniformBuffer:c,simBindGroup:e.createBindGroup({layout:r,entries:[{binding:0,resource:{buffer:l}}]}),simHighlightBindGroup:e.createBindGroup({layout:r,entries:[{binding:0,resource:{buffer:c}}]})}}const Rp=1e30,Js=Ct.cullMargin,Bp=Number.POSITIVE_INFINITY,Qs=Ct.slotBytes;function Fp(e,t,n=Js){const i=new Float32Array(Math.max(e.buffer_bytes,Qs)/4);for(const r of e.parameters){const a=t?.[r.name];let s=a&&a.length===r.components?a:r.value;t&&r.depends?.some(o=>o in t)&&r.name.endsWith(".half")&&(s=s.map(()=>Rp));for(let o=0;o<r.components;o+=1){const l=s[o];i[r.offset/4+o]=l??Number.NaN}}return e.nan_offset!==void 0&&(i[e.nan_offset/4]=Number.NaN),e.cull_margin_offset!=null&&(i[e.cull_margin_offset/4]=n),i}function zp(e,t){return!e||!t?e===t||!e&&!t:e.group!==t.group||e.binding!==t.binding||e.buffer_bytes!==t.buffer_bytes||e.parameters.length!==t.parameters.length?!1:e.parameters.every((n,i)=>{const r=t.parameters[i];return n.name===r.name&&n.offset===r.offset&&n.components===r.components})}class Np{constructor(t=8){this.capacity=t}entries=new Map;hits=0;misses=0;async get(t,n,i){const r=this.entries.get(n);if(r)return this.entries.delete(n),this.entries.set(n,r),this.hits+=1,r;this.misses+=1;const a=t.createShaderModule({code:n,label:i}),o=(await a.getCompilationInfo()).messages.filter(l=>l.type==="error");if(o.length)throw new Error(o.map(l=>`${i} ${l.lineNum}:${l.linePos} ${l.message}`).join(`
`));for(this.entries.set(n,a);this.entries.size>this.capacity;){const l=this.entries.keys().next();if(l.done)break;this.entries.delete(l.value)}return a}get size(){return this.entries.size}clear(){this.entries.clear()}}class Gp{constructor(t,n){this.host=t,this.uploadedCullBounds=n}shaderModules=new Np;_installedShaders=null;_installedPreview=null;program=null;parameterBuffer=null;parameterBufferBytes=0;previewParameterGroup=null;pathParameterGroup=null;parameterOverrides=null;uploadedCullBounds;revision=0;_previewPipeline=null;_previewDepthPipeline=null;_previewBindGroup=null;_pathPipeline=null;_presentPipeline=null;_pathReady=!1;_pipelineBuilds=0;_parameterUploads=0;get installedShaders(){return this._installedShaders}get installedPreview(){return this._installedPreview}get previewPipeline(){return this._previewPipeline}get previewDepthPipeline(){return this._previewDepthPipeline}get previewBindGroup(){return this._previewBindGroup}get pathPipeline(){return this._pathPipeline}get presentPipeline(){return this._presentPipeline}get pathReady(){return this._pathReady}get pipelineBuilds(){return this._pipelineBuilds}get parameterUploads(){return this._parameterUploads}get moduleStats(){return{hits:this.shaderModules.hits,misses:this.shaderModules.misses}}async setShaders(t,n){const i=n.program??null;if(this.isValuesOnlyEdit(n,i)){this.program=i,this.parameterOverrides=null,this.uploadParameters(),this._parameterUploads+=1,this.host.sceneChanged();return}const r=++this.revision;this._pathReady=!1;const a=this.host.format(),s=await this.shaderModules.get(t,n.preview,"Preview WGSL"),o=si(t,"Preview bindings",[0,2]),l=this.parameterLayoutFor(i),c=t.createPipelineLayout({bindGroupLayouts:this.sceneBindGroupLayouts(o.bindGroupLayout,i,l)}),d=y=>({layout:c,vertex:{module:s,entryPoint:"vs_main"},fragment:{module:s,entryPoint:"fs_main_depth",targets:[{format:a,writeMask:y}]},primitive:{topology:"triangle-list"},depthStencil:{format:He,depthWriteEnabled:!0,depthCompare:"always"}}),[h,u]=await Promise.all([t.createRenderPipelineAsync(d(GPUColorWrite.ALL)),t.createRenderPipelineAsync(d(0))]);if(this._pipelineBuilds+=2,r!==this.revision)return;this.program=i,this.parameterOverrides=null,this.adoptParameterBuffer(),this._previewPipeline=h,this._installedPreview=n.preview,this.host.previewReplaced(),this._previewDepthPipeline=u;const p=this.host.uniformBuffers();this._previewBindGroup=t.createBindGroup({layout:o.bindGroupLayout,entries:[{binding:0,resource:{buffer:p.scene}},{binding:2,resource:{buffer:p.view}}]}),this.previewParameterGroup=this.parameterBindGroup(l),this.pathParameterGroup=null,this.uploadParameters(),this.host.sceneChanged();const[f,m]=await Promise.all([this.shaderModules.get(t,n.path,"Path tracer WGSL"),this.shaderModules.get(t,n.present,"Present WGSL")]),[w,x]=await Promise.all([t.createRenderPipelineAsync({layout:"auto",vertex:{module:f,entryPoint:"vs_main"},fragment:{module:f,entryPoint:"fs_path_trace",targets:[{format:"rgba16float"}]},primitive:{topology:"triangle-list"}}),t.createRenderPipelineAsync({layout:"auto",vertex:{module:m,entryPoint:"vs_present"},fragment:{module:m,entryPoint:"fs_present",targets:[{format:a}]},primitive:{topology:"triangle-list"},depthStencil:{format:He,depthWriteEnabled:!1,depthCompare:"always"}})]);this._pipelineBuilds+=2,r===this.revision&&(this._pathPipeline=w,this._presentPipeline=x,this.pathParameterGroup=this.program?this.parameterBindGroup(w.getBindGroupLayout(this.program.group)):null,this._installedShaders={...n},this._pathReady=!0,this.host.invalidate())}bindParameters(t,n){const i=n==="preview"?this.previewParameterGroup:this.pathParameterGroup;i&&t.setBindGroup(this.program.group,i)}isValuesOnlyEdit(t,n){const i=this._installedShaders;return!!(i&&this._previewPipeline&&this._pathPipeline&&t.preview===i.preview&&t.path===i.path&&t.present===i.present&&zp(n,this.program))}parameterLayoutFor(t){const n=this.host.device();return!n||!t?null:n.createBindGroupLayout({label:"SDF parameter bindings",entries:[{binding:t.binding,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]})}sceneBindGroupLayouts(t,n,i){const r=this.host.device(),a=[t];if(!r||!n||!i)return a;for(;a.length<n.group;)a.push(r.createBindGroupLayout({entries:[]}));return a.push(i),a}parameterBindGroup(t){const n=this.host.device();return!n||!this.program||!this.parameterBuffer||!t?null:n.createBindGroup({label:"SDF parameters",layout:t,entries:[{binding:this.program.binding,resource:{buffer:this.parameterBuffer}}]})}adoptParameterBuffer(){const t=this.host.device();if(!t)return;const n=this.program?Math.max(this.program.buffer_bytes,Qs):0;n!==this.parameterBufferBytes&&(this.parameterBufferBytes=n,this.parameterBuffer=n?t.createBuffer({label:"SDF parameters",size:n,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}):null)}uploadParameters(){const t=this.host.device();if(!t||!this.program||!this.parameterBuffer)return;const n=this.host.cullBounds();this.uploadedCullBounds=n;const i=Fp(this.program,this.parameterOverrides??void 0,n?Js:Bp);t.queue.writeBuffer(this.parameterBuffer,0,i.buffer)}syncCullMargin(){!this.program||this.uploadedCullBounds===this.host.cullBounds()||this.uploadParameters()}setParameterOverrides(t){return!this.program||!this.parameterBuffer?!1:(this.parameterOverrides=t?{...t}:null,this.uploadParameters(),this._parameterUploads+=1,this.host.sceneChanged(),!0)}get parameterProgram(){return this.program}dropStaleParameterOverrides(){this.parameterOverrides&&this.setParameterOverrides(null)}get hasParameterBuffer(){return this.program!==null&&this.parameterBuffer!==null}}function jp(e){const{size:t,camera:n,position:i,display:r,quality:a}=e,{sdfView:s,sdfAxis:o,sdfFraction:l,isoOffset:c}=r,d=vl[s]??0;return new Float32Array([t.width,t.height,d,o,i[0],i[1],i[2],yr(l),n.target[0],n.target[1],n.target[2],c,.55,.35,.8,_p,Di[0],Di[1],Di[2],Is(n.distance),e.sampleCount,a.bounces,a.shadowSamples,es(r,a),r.projection==="orthographic"?1:0,Dt(n.distance),e.flags,r.xray])}function Ia(e,t,n,i,r,a){const s=new Float32Array(fr);return s.set(e,0),s.set([t[0],t[1],t[2],0],16),s.set([n.width,n.height,xp,Sp],20),s.set([i,r,0,0],24),s.set([a.forward[0],a.forward[1],a.forward[2],a.orthographic?1:0],28),s}const fr=32;function Aa(e){const{position:t,clip:n}=e,{normal:i,offset:r}=Dh(n,e.bounds),[a,s]=e.range,o=s-a,l=o>1e-12?1/o:0,c=new Float32Array(28);return c.set(e.matrix,0),c.set([i[0],i[1],i[2],r],16),c.set([a,l,e.pixel,n.enabled?1:0],20),c.set([e.ramp==="quality"?1:0,t[0],t[1],t[2]],24),c}function Vp(e){const{size:t,camera:n,position:i,emphasis:r}=e,a=t.width/Math.max(e.cssWidth,1),s=_=>_*a,o=_=>le(se[_]),[l,c,d]=o("graticule-line"),[h,u,p]=o("graticule-axis"),{forward:f,right:m,up:w}=qt(i,n.target),x=n.distance,y=n.target;return new Float32Array([t.width,t.height,Math.max(1,s($a.lineWidth)),e.projection==="orthographic"?1:0,i[0],i[1],i[2],Dt(x),m[0],m[1],m[2],t.width/Math.max(t.height,1),w[0],w[1],w[2],Is(x),f[0],f[1],f[2],Tt,l,c,d,Tn.minor*r,l,c,d,Tn.major*r,h,u,p,Tn.axis*r,x*Jr.start,x*Jr.end,$a.axisWidth,ks,y[0],y[1],y[2],0])}class Hp{active=!1;edgesVisible=!1;ramp="field";pipelines=null;vertexBuffer=null;indexBuffer=null;indexCount=0;range=[0,0];bounds={min:[0,0,0],max:[0,0,0]};highlight=null;outlineBuffer=null;outlineCount=0;clip={...Oh};payload=null;scalarOverride=null;positionOverride=null;overlay=null;edgeIndexBuffer=null;edgeIndexCount=0;attach(t){this.pipelines=t}setMesh(t,n){if(this.indexCount=0,this.highlight=null,this.outlineBuffer?.destroy(),this.outlineBuffer=null,this.outlineCount=0,this.payload=n,this.scalarOverride=null,this.positionOverride=null,this.overlay=null,!n||!t||n.indices.length===0)return;const i=new Uint32Array(n.indices);if(this.indexBuffer?.destroy(),this.indexBuffer=t.createBuffer({label:"simulation indices",size:i.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.indexBuffer,0,i),this.indexCount=i.length,this.edgeIndexBuffer?.destroy(),this.edgeIndexBuffer=null,this.edgeIndexCount=0,n.edges&&n.edges.length>=2){const r=new Uint32Array(n.edges);this.edgeIndexBuffer=t.createBuffer({label:"simulation element edges",size:r.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.edgeIndexBuffer,0,r),this.edgeIndexCount=r.length}this.range=n.range,this.uploadVertices(t)}setScalars(t,n,i){this.scalarOverride=n,i?this.range=i:!n&&this.payload&&(this.range=this.payload.range),this.uploadVertices(t)}setPositions(t,n){this.positionOverride=n,this.uploadVertices(t)}setOverlay(t,n){this.overlay=n,this.uploadVertices(t)}setHighlight(t,n){this.highlight=n,this.outlineBuffer?.destroy(),this.outlineBuffer=null,this.outlineCount=0;const i=this.payload?.indices;if(n&&i&&t){const r=Rh(i,n.start,n.count);r.length>=2&&(this.outlineBuffer=t.createBuffer({label:"simulation hover outline",size:r.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.outlineBuffer,0,r),this.outlineCount=r.length)}}setClip(t){this.clip={...t}}uploadVertices(t){const n=this.payload;if(!n||!t||this.indexCount===0)return;const i=this.positionOverride??n.positions,r=this.scalarOverride??n.scalars,a=this.overlay,s=n.vertex_count,o=new Float32Array(s*8);for(let l=0;l<s;l++){const c=l*8;o[c]=i[l*3],o[c+1]=i[l*3+1],o[c+2]=i[l*3+2],o[c+3]=r[l]??0,a&&(o[c+4]=a[l*4],o[c+5]=a[l*4+1],o[c+6]=a[l*4+2],o[c+7]=a[l*4+3])}this.vertexBuffer?.destroy(),this.vertexBuffer=t.createBuffer({label:"simulation vertices",size:o.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.vertexBuffer,0,o),this.bounds=Lh(i)}writeUniforms(t,n,i,r){if(!this.active||!this.pipelines)return;const a={matrix:n,position:i,clip:this.clip,bounds:this.bounds,range:this.range,ramp:this.ramp};t.queue.writeBuffer(this.pipelines.simUniformBuffer,0,Aa({...a,pixel:0})),t.queue.writeBuffer(this.pipelines.simHighlightUniformBuffer,0,Aa({...a,pixel:2/Math.max(r,1)}))}draw(t,n){const i=this.pipelines;!this.active||!i||!this.vertexBuffer||!this.indexBuffer||this.indexCount===0||(t.setPipeline(i.simPipeline),t.setBindGroup(0,i.simBindGroup),t.setVertexBuffer(0,this.vertexBuffer),t.setIndexBuffer(this.indexBuffer,"uint32"),t.drawIndexed(this.indexCount),this.edgesVisible&&this.edgeIndexBuffer&&this.edgeIndexCount>0&&(t.setPipeline(i.simEdgePipeline),t.setBindGroup(0,i.simBindGroup),t.setVertexBuffer(0,this.vertexBuffer),t.setIndexBuffer(this.edgeIndexBuffer,"uint32"),t.drawIndexed(this.edgeIndexCount)),n&&this.highlight&&this.outlineBuffer&&this.outlineCount>0&&(t.setPipeline(i.simEdgePipeline),t.setBindGroup(0,i.simHighlightBindGroup),t.setVertexBuffer(0,this.vertexBuffer),t.setIndexBuffer(this.outlineBuffer,"uint32"),t.drawIndexed(this.outlineCount,3)))}destroy(){this.vertexBuffer?.destroy(),this.outlineBuffer?.destroy(),this.indexBuffer?.destroy(),this.edgeIndexBuffer?.destroy()}}class Up{constructor(t={}){this.callbacks=t}canvas;device=null;context=null;format="bgra8unorm";adapterLabel="WebGPU";uniformBuffer;viewBuffer;overlayBuffer;meshOverlayBuffer;graticuleBuffer;scene=new Gp({device:()=>this.device,format:()=>this.format,uniformBuffers:()=>({scene:this.uniformBuffer,view:this.viewBuffer}),cullBounds:()=>this.display.cullBounds,previewReplaced:()=>{this.previewMsPerPixel=null},sceneChanged:()=>{this.destroyAccumulation(),this.invalidate()},invalidate:()=>this.invalidate()},jt.cullBounds);groundEmphasis=1;graticulePipeline=null;graticuleBindGroup=null;depthTexture=null;accumulation=[];pathBindGroups=[];presentBindGroups=[];accumulationWidth=0;accumulationHeight=0;readIndex=0;sampleCount=0;overlay=new Pp;brep=new wp;simulation=new Hp;framePending=!1;initError="";camera={...xt.iso,distance:4.6,target:[0,0,0]};display={...jt};quality=Ze[Pn];pathTracing=!1;_interacting=!1;previewMsPerPixel=null;gestureScale=1;upscalePipeline=null;gestureColor=null;gestureDepth=null;gestureBindGroup=null;get interacting(){return this._interacting}set interacting(t){t!==this._interacting&&(this._interacting=t,this.gestureScale=t?this.chooseGestureScale():1,t||this.invalidate())}chooseGestureScale(){if(!this.canvas||this.previewMsPerPixel===null)return 1;const t=this.previewMsPerPixel*this.canvas.width*this.canvas.height;if(t<=Pa)return 1;const n=Math.sqrt(Pa/t);return Math.max($p,Math.floor(n*8)/8)}recordPreviewCost(t,n){const i=t/Math.max(n,1);this.previewMsPerPixel=this.previewMsPerPixel===null?i:this.previewMsPerPixel*.7+i*.3}ensureGestureTargets(t){const n=this.device,i=Math.max(1,Math.ceil(this.canvas.width*t)),r=Math.max(1,Math.ceil(this.canvas.height*t));return(this.gestureColor?.width!==i||this.gestureColor?.height!==r)&&(this.gestureColor?.destroy(),this.gestureDepth?.destroy(),this.gestureColor=n.createTexture({label:"gesture preview colour",size:[i,r],format:this.format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING}),this.gestureDepth=n.createTexture({label:"gesture preview depth",size:[i,r],format:He,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING}),this.upscalePipeline??=Op(n,this.format),this.gestureBindGroup=n.createBindGroup({layout:this.upscalePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:this.gestureColor.createView()},{binding:1,resource:this.gestureDepth.createView()},{binding:2,resource:n.createSampler({magFilter:"linear",minFilter:"linear"})}]})),{width:i,height:r}}get unavailableReason(){return this.initError}get gizmoMode(){return this.overlay.gizmoMode}set gizmoMode(t){t!==this.overlay.gizmoMode&&(this.overlay.gizmoMode=t,this.uploadOverlay(),this.invalidate())}get gizmoAxis(){return this.overlay.gizmoAxis}set gizmoAxis(t){t!==this.overlay.gizmoAxis&&(this.overlay.gizmoAxis=t,this.uploadOverlay(),this.scheduleRender())}gizmoModeFor(t){return this.overlay.gizmoModeFor(t)}gizmoTarget(){return this.overlay.gizmoTarget()}get viewport(){return{width:this.canvas?.width??1,height:this.canvas?.height??1}}get view(){return{position:Gi(this.camera),target:this.camera.target,width:this.viewport.width,height:this.viewport.height,projection:this.display.projection,orthoHeight:Dt(this.camera.distance)}}applyViewPreset(t){const n=xt[t];n&&(this.camera={...this.camera,yaw:n.yaw,pitch:n.pitch},this.invalidate())}displayFlags(){return yl(this.display,this.simulation.active||this.brepActive)}get brepActive(){return this.display.showBrep===!0&&this.brep.hasFaces&&!this.simulation.active}setBrep(t){this.device&&(this.brep.set(this.device,t),this.invalidate())}async init(t){if(this.canvas=t,!window.isSecureContext){this.fail("WebGPU requires a secure context. Open this viewer on localhost or over HTTPS.");return}if(!navigator.gpu){this.fail("WebGPU is not available in this browser. Use a current Chrome, Edge, or Safari.");return}try{const n=await Ea(navigator.gpu.requestAdapter({powerPreference:"high-performance"}),8e3,"WebGPU adapter request");if(!n){this.fail("No WebGPU adapter was found on this system.");return}this.adapterLabel=await this.identify(n),this.device=await Ea(n.requestDevice(),8e3,"WebGPU device request");const i=t.getContext("webgpu");if(!i){this.fail("The browser exposed WebGPU but could not create a canvas context.");return}this.context=i,this.format=navigator.gpu.getPreferredCanvasFormat(),this.context.configure({device:this.device,format:this.format,alphaMode:"opaque"}),this.uniformBuffer=this.createUniform(112),this.viewBuffer=this.createUniform(64),this.overlayBuffer=this.createUniform(fr*4),this.meshOverlayBuffer=this.createUniform(fr*4),this.graticuleBuffer=this.createUniform(Tp),this.buildPipelines(),this.device.addEventListener("uncapturederror",r=>{const a=r.error?.message??"WebGPU error";this.callbacks.onError?.(`WebGPU validation error: ${a}`)}),this.device.lost.then(r=>{this.initError=`WebGPU device lost: ${r.message}`,this.callbacks.onError?.(this.initError)}),this.callbacks.onReady?.()}catch(n){this.fail(n instanceof Error?n.message:String(n))}}fail(t){this.initError=t,this.callbacks.onError?.(t)}async identify(t){try{const n=t.info??await t.requestAdapterInfo?.(),i=[n?.vendor,n?.architecture].filter(Boolean);return i.length?`WebGPU · ${i.join(" ")}`:"WebGPU"}catch{return"WebGPU"}}createUniform(t){return this.device.createBuffer({size:t,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}buildPipelines(){const t=this.device,n=Cp(t,this.format,this.overlayBuffer,this.meshOverlayBuffer);this.overlay.attach(n),this.brep.attach(n);const i=Lp(t,this.format,this.graticuleBuffer);this.graticulePipeline=i.graticulePipeline,this.graticuleBindGroup=i.graticuleBindGroup,this.simulation.attach(Dp(t,this.format,r=>this.createUniform(r))),this.invalidate()}get simulationActive(){return this.simulation.active}set simulationActive(t){t!==this.simulation.active&&(this.simulation.active=t,this.uploadOverlay(),this.invalidate())}setSimulationMesh(t){this.simulation.setMesh(this.device,t),this.invalidate()}get simulationEdgesVisible(){return this.simulation.edgesVisible}set simulationEdgesVisible(t){t!==this.simulation.edgesVisible&&(this.simulation.edgesVisible=t,this.invalidate())}get simulationRamp(){return this.simulation.ramp}set simulationRamp(t){this.simulation.ramp=t}setSimulationScalars(t,n){this.simulation.setScalars(this.device,t,n),this.invalidate()}setSimulationPositions(t){this.simulation.setPositions(this.device,t),this.invalidate()}setSimulationOverlay(t){this.simulation.setOverlay(this.device,t),this.invalidate()}setSimulationHighlight(t){this.simulation.setHighlight(this.device,t),this.scheduleRender()}setSimulationClip(t){this.simulation.setClip(t),this.scheduleRender()}setShaders(t){return this.device?this.scene.setShaders(this.device,t):(this.callbacks.onError?.(this.initError||"WGSL compiled, but WebGPU is unavailable."),Promise.resolve())}get pathReady(){return this.scene.pathReady}get pipelineBuilds(){return this.scene.pipelineBuilds}get parameterUploads(){return this.scene.parameterUploads}setParameterOverrides(t){return this.scene.setParameterOverrides(t)}get parameterProgram(){return this.scene.parameterProgram}dropStaleParameterOverrides(){this.scene.dropStaleParameterOverrides()}get hasParameterBuffer(){return this.scene.hasParameterBuffer}get shaderStats(){return{pipelineBuilds:this.scene.pipelineBuilds,parameterUploads:this.scene.parameterUploads,hasParameterBuffer:this.scene.hasParameterBuffer,...this.scene.moduleStats}}setConstruction(t,n,i,r=ls(t)){this.overlay.setConstruction(t,n,i,r),this.refreshOverlay()}setFocusGroup(t){this.overlay.setFocusGroup(t)&&this.refreshOverlay()}setMeshEdges(t){this.overlay.setMeshEdges(t),this.refreshOverlay()}setPlanePreview(t){this.overlay.setPlanePreview(t)&&this.refreshOverlay()}setFaceHighlight(t){this.overlay.setFaceHighlight(t)&&this.refreshOverlay()}uploadOverlay(){this.device&&this.overlay.upload(this.device,this.simulation.active,this.view)}refreshOverlay(){this.uploadOverlay(),this.scheduleRender()}destroyAccumulation(){for(const t of this.accumulation)t.destroy();this.accumulation=[],this.pathBindGroups=[],this.presentBindGroups=[],this.accumulationWidth=0,this.accumulationHeight=0}ensureAccumulation(){if(this.accumulation.length===2&&this.accumulationWidth===this.canvas.width&&this.accumulationHeight===this.canvas.height)return;this.destroyAccumulation();const t=this.device;this.accumulationWidth=this.canvas.width,this.accumulationHeight=this.canvas.height;const n={size:[this.canvas.width,this.canvas.height],format:"rgba16float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT};this.accumulation=[t.createTexture(n),t.createTexture(n)],this.pathBindGroups=this.accumulation.map(i=>t.createBindGroup({layout:this.scene.pathPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:i.createView()}]})),this.presentBindGroups=this.accumulation.map(i=>t.createBindGroup({layout:this.scene.presentPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:i.createView()}]})),this.readIndex=0,this.sampleCount=0}ensureDepthTexture(){return(!this.depthTexture||this.depthTexture.width!==this.canvas.width||this.depthTexture.height!==this.canvas.height)&&(this.depthTexture?.destroy(),this.depthTexture=this.device.createTexture({label:"scene depth",size:[this.canvas.width,this.canvas.height],format:He,usage:GPUTextureUsage.RENDER_ATTACHMENT})),this.depthTexture.createView()}resize(){if(!this.canvas)return;const t=Math.max(1,this.canvas.clientWidth),n=Math.max(1,this.canvas.clientHeight),i=Math.min(window.devicePixelRatio||1,this.quality.maxRatio),r=Math.sqrt(this.quality.pixelBudget/(t*n)),a=Math.min(i,r),s=Math.max(1,Math.floor(t*a)),o=Math.max(1,Math.floor(n*a));this.canvas.width===s&&this.canvas.height===o||(this.canvas.width=s,this.canvas.height=o,this.device&&this.context&&(this.context.configure({device:this.device,format:this.format,alphaMode:"opaque"}),this.destroyAccumulation()))}reconfigure(){!this.device||!this.context||this.context.configure({device:this.device,format:this.format,alphaMode:"opaque"})}invalidate(){this.sampleCount=0,this.scheduleRender()}scheduleRender(){this.framePending||(this.framePending=!0,requestAnimationFrame(()=>this.render()))}writeUniforms(){const t=this.device;this.scene.syncCullMargin();const n=Gi(this.camera),i={width:this.canvas.width,height:this.canvas.height};t.queue.writeBuffer(this.uniformBuffer,0,jp({size:i,camera:this.camera,position:n,display:this.display,quality:this.quality,sampleCount:this.sampleCount,flags:this.displayFlags()}));const r=Xo(this.view);t.queue.writeBuffer(this.viewBuffer,0,r);const a={forward:qt(n,this.camera.target).forward,orthographic:this.display.projection==="orthographic"};t.queue.writeBuffer(this.overlayBuffer,0,Ia(r,n,i,Sa,kp,a)),t.queue.writeBuffer(this.meshOverlayBuffer,0,Ia(r,n,i,Sa*.4,0,a)),this.display.showGraticule&&t.queue.writeBuffer(this.graticuleBuffer,0,Vp({size:i,cssWidth:this.canvas.clientWidth,camera:this.camera,position:n,projection:this.display.projection,emphasis:this.groundEmphasis})),this.simulation.writeUniforms(t,r,n,this.canvas.width)}drawGraticule(t){!this.display.showGraticule||!this.graticulePipeline||!this.graticuleBindGroup||(t.setPipeline(this.graticulePipeline),t.setBindGroup(0,this.graticuleBindGroup),t.draw(3))}get seeThroughOverlays(){return this.display.xray>0&&!this.brepActive}drawOverlay(t){this.overlay.draw(t,this.display,this.seeThroughOverlays)}renderEmpty(){const t=this.device;this.writeUniforms();const n=t.createCommandEncoder(),i=n.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:wn,loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.ensureDepthTexture(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});this.drawGraticule(i),i.end(),t.queue.submit([n.finish()])}frameLog=[];cpuLog=[];get frameStats(){const t=performance.now(),n=this.frameLog.filter(r=>t-r<=1e3),i=this.cpuLog.length?this.cpuLog.reduce((r,a)=>r+a,0)/this.cpuLog.length:null;return{fps:n.length?n.length:null,cpuMs:n.length?i:null,width:this.canvas?.width??0,height:this.canvas?.height??0}}render(){const t=performance.now();this.renderFrame();const n=performance.now();this.frameLog.push(n),this.frameLog.length>240&&this.frameLog.splice(0,this.frameLog.length-240),this.cpuLog.push(n-t),this.cpuLog.length>30&&this.cpuLog.shift()}renderFrame(){if(this.framePending=!1,!this.device||!this.context)return;this.resize();const t=this.scene,{previewPipeline:n,previewBindGroup:i}=t;if(!n||!i){this.renderEmpty();return}this.writeUniforms();const r=this.device,a=r.createCommandEncoder(),s=this.context.getCurrentTexture().createView(),l={view:this.ensureDepthTexture(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"},c=this.pathTracing&&t.pathReady&&!this.interacting&&!this.simulation.active&&!this.brepActive;let d=0;if(c){this.ensureAccumulation();const h=1-this.readIndex,u=a.beginRenderPass({colorAttachments:[{view:this.accumulation[h].createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});u.setPipeline(t.pathPipeline),u.setBindGroup(0,this.pathBindGroups[this.readIndex]),t.bindParameters(u,"path"),u.draw(3),u.end();const p=a.beginRenderPass({colorAttachments:[{view:s,clearValue:wn,loadOp:"clear",storeOp:"store"}],depthStencilAttachment:l});p.setPipeline(t.presentPipeline),p.setBindGroup(0,this.presentBindGroups[h]),p.draw(3),(this.overlay.hasSkeletonEdges&&this.display.showSketches||this.display.showGraticule)&&t.previewDepthPipeline&&(p.setPipeline(t.previewDepthPipeline),p.setBindGroup(0,i),t.bindParameters(p,"preview"),p.draw(3)),this.drawGraticule(p),this.drawOverlay(p),p.end(),this.readIndex=h}else{const h=this.simulation.active||this.brepActive?1:this.gestureScale;if(h<1){const p=this.ensureGestureTargets(h);r.queue.writeBuffer(this.uniformBuffer,0,new Float32Array([p.width,p.height]));const f=a.beginRenderPass({colorAttachments:[{view:this.gestureColor.createView(),clearValue:wn,loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.gestureDepth.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});f.setPipeline(n),f.setBindGroup(0,i),t.bindParameters(f,"preview"),f.draw(3),f.end()}d=h<1?this.gestureColor.width*this.gestureColor.height:this.canvas.width*this.canvas.height;const u=a.beginRenderPass({colorAttachments:[{view:s,clearValue:wn,loadOp:"clear",storeOp:"store"}],depthStencilAttachment:l});h<1?(u.setPipeline(this.upscalePipeline),u.setBindGroup(0,this.gestureBindGroup)):(u.setPipeline(n),u.setBindGroup(0,i),t.bindParameters(u,"preview")),u.draw(3),this.simulation.draw(u,this.display.showOverlays),this.brepActive&&this.brep.draw(u),this.drawGraticule(u),this.drawOverlay(u),u.end()}if(r.queue.submit([a.finish()]),d>0){const h=performance.now();r.queue.onSubmittedWorkDone().then(()=>this.recordPreviewCost(performance.now()-h,d))}if(c)this.sampleCount+=1,this.callbacks.onStatus?.("ready",`${this.quality.label} path trace · ${this.adapterLabel} · ${this.sampleCount} spp`),this.sampleCount<this.quality.samples&&this.scheduleRender();else{const h=this.pathTracing?t.pathReady?" · moving":" · preparing path tracer…":"";this.callbacks.onStatus?.("ready",`${this.quality.label} preview · ${this.adapterLabel}${h}`)}}async benchmark(t=30,n=!1){const i=this.device;if(!i||!this.scene.previewPipeline)return{medianMs:NaN,meanMs:NaN,samples:[],width:0,height:0};const r=this.interacting,a=this.camera.yaw;this.interacting=n;const s=[];let o=0,l=0;for(let d=-2;d<t;d++){this.camera={...this.camera,yaw:a+(d+2)*.01},this.sampleCount=0;const h=performance.now();this.renderFrame(),await i.queue.onSubmittedWorkDone(),d>=0&&s.push(performance.now()-h),o=this.canvas.width,l=this.canvas.height}this.interacting=r,this.camera={...this.camera,yaw:a},this.invalidate();const c=[...s].sort((d,h)=>d-h);return{medianMs:c[Math.floor(c.length/2)],meanMs:s.reduce((d,h)=>d+h,0)/s.length,samples:s,width:o,height:l}}async benchmarkCompile(){const t=this.scene.installedShaders,n=this.device;if(!t||!n)return{firstFrameMs:NaN,pathReadyMs:NaN};const i=`
// bench ${performance.now()} ${Math.random()}
`,r=performance.now(),a=this.setShaders({...t,preview:t.preview+i,path:t.path+i,present:t.present+i});let s=NaN;const o=(async()=>{for(;this.scene.installedPreview!==t.preview+i;)await new Promise(c=>requestAnimationFrame(c));this.renderFrame(),await n.queue.onSubmittedWorkDone(),s=performance.now()-r})();await a;const l=performance.now()-r;return await o,{firstFrameMs:s,pathReadyMs:l}}destroy(){this.destroyAccumulation(),this.gestureColor?.destroy(),this.gestureDepth?.destroy(),this.depthTexture?.destroy(),this.overlay.destroy(),this.brep.destroy(),this.simulation.destroy(),this.device?.destroy()}}const qp=`{"ok":true,"installed":false,"provider":"diff-brep","contract_version":1,"kinds":{"node_map":{"kind":"node_map","available":false,"plugin":null,"version":null,"contract_version":null,"compatible":false,"reason":"not registered","label":"Frozen geometry \\u00b7 public tier","message":"A Gmsh mesh is frozen geometry in the public tier: its nodes do not follow the design, because the node map is part of the private tier (diff-brep). Use mesher='tetgen' for a mesh that follows the design, or install diff-brep."},"feature_edges":{"kind":"feature_edges","available":false,"plugin":null,"version":null,"contract_version":null,"compatible":false,"reason":"not registered","label":"Approximate \\u00b7 lattice","message":"Feature edges are read off the dual-contouring lattice in the public tier, so they are approximate; exact edge curves are part of the private tier (diff-brep)."},"brep":{"kind":"brep","available":false,"plugin":null,"version":null,"contract_version":null,"compatible":false,"reason":"not registered","label":"Dual contouring \\u00b7 public tier","message":"Exact B-rep extraction is part of the private tier (diff-brep); the public tier has the dual-contour mesh only."},"step_export":{"kind":"step_export","available":false,"plugin":null,"version":null,"contract_version":null,"compatible":false,"reason":"not registered","label":"Faceted \\u00b7 public tier","message":"STEP export is faceted in the public tier: triangles from dual contouring. Exact planes, cylinders and edge curves are part of the private tier (diff-brep)."},"drag":{"kind":"drag","available":false,"plugin":null,"version":null,"contract_version":null,"compatible":false,"reason":"not registered","label":"Unavailable \\u00b7 public tier","message":"Dragging a B-rep handle solves the design through the exact B-rep, which is part of the private tier (diff-brep)."}}}
`,eo=Object.assign({"./bundle/bracket.demo.json":()=>tt(()=>import("./bracket.demo-I140dy0n.js"),[],import.meta.url).then(e=>e.default),"./bundle/cylinder_block.demo.json":()=>tt(()=>import("./cylinder_block.demo-BWqejYZZ.js"),[],import.meta.url).then(e=>e.default),"./bundle/duct_fairing.demo.json":()=>tt(()=>import("./duct_fairing.demo-DkGBTTak.js"),[],import.meta.url).then(e=>e.default),"./bundle/duct_sink.demo.json":()=>tt(()=>import("./duct_sink.demo-kh5kVN8F.js"),[],import.meta.url).then(e=>e.default),"./bundle/end_cap.demo.json":()=>tt(()=>import("./end_cap.demo-BPE0fSn2.js"),[],import.meta.url).then(e=>e.default),"./bundle/motor_shield.demo.json":()=>tt(()=>import("./motor_shield.demo-BOBLyZ19.js"),[],import.meta.url).then(e=>e.default),"./bundle/starter.demo.json":()=>tt(()=>import("./starter.demo-DYwTW_bI.js"),[],import.meta.url).then(e=>e.default)}),to=e=>e.split("/").pop().replace(".demo.json",""),_n=Object.keys(eo).map(to).sort(),Wp=JSON.parse(qp);function Yp(e){const t=JSON.parse(e);if(t.format!==vn.format||t.version!==vn.version)throw new Error(`scene file is ${t.format} v${t.version}; this page reads ${vn.format} v${vn.version}`);return t}async function Xp(e){const t=Object.entries(eo).find(([n])=>to(n)===e)?.[1];if(!t)throw new Error(`no shipped scene named ${e}`);return Yp(await t())}const Kp=35;function Zp(e,t,n,i=Kp){const r=h=>[e[3*h],e[3*h+1],e[3*h+2]],a=[],s=new Map,o=e.length/3;for(let h=0;h<t.length/3;h++){const[u,p,f]=[t[3*h],t[3*h+1],t[3*h+2]],[m,w,x]=[r(u),r(p),r(f)],y=[w[0]-m[0],w[1]-m[1],w[2]-m[2]],_=[x[0]-m[0],x[1]-m[1],x[2]-m[2]],$=[y[1]*_[2]-y[2]*_[1],y[2]*_[0]-y[0]*_[2],y[0]*_[1]-y[1]*_[0]],E=Math.hypot(...$)||1;a.push([$[0]/E,$[1]/E,$[2]/E]);for(const[I,A]of[[u,p],[p,f],[f,u]]){const B=Math.min(I,A)*o+Math.max(I,A),N=s.get(B);N?N.push(h):s.set(B,[h])}}const l=Math.cos(i*Math.PI/180),c=[],d=[];for(const[h,u]of s){if(u.length===2&&u[0]%2===0&&u[1]===u[0]+1)continue;const p=[r(Math.floor(h/o)),r(h%o)];if(c.push(p),u.length!==2)continue;const[f,m]=[a[u[0]],a[u[1]]];f[0]*m[0]+f[1]*m[1]+f[2]*m[2]<l&&d.push(p)}return{wire:c,sharp:d,resolution:n,edges:"lattice"}}function Ma(e){return Object.fromEntries(e.map(t=>[t.name,[...t.value]]))}const Ht=e=>e.drivenBy===null;function no(e,t,n){return Math.min(e.upper[t],Math.max(e.lower[t],n))}function Jp(e,t,n){const i=Object.fromEntries(Object.entries(t).map(([r,a])=>[r,[...a]]));for(const r of e){const a=n[r.name];!a||!Ht(r)||a.length!==r.components||a.some(s=>!Number.isFinite(s))||(i[r.name]=a.map((s,o)=>no(r,o,s)))}return i}function Qp(e,t){const n={};for(const i of e){const r=t[i.name];r&&r.some((a,s)=>a!==i.value[s])&&(n[i.name]=[...r])}return n}function io(e,t){const n={...e.params};for(const[i,r]of Object.entries(e.free??{}))if(Array.isArray(r)){const a=n[i].map((s,o)=>{const l=r[o],c=l?t[l]:void 0;return c?[...c]:s});n[i]=a}else{const a=t[r];a&&(n[i]=a.length===1?a[0]:[...a])}return{...e,params:n,children:e.children.map(i=>io(i,t))}}function Ca(e,t){return{...e,root:io(e.root,t)}}const ro=e=>new Map(e.map(t=>[t.name,t]));function ao(e,t){const n=t[e.name];return!n||n.length!==e.components?null:[...n]}function Ri(e,t,n){if(!e?.length)return null;const i=typeof t=="number"?1:t.length,r=typeof t=="number"?[t]:[...t];for(const a of e){const s=ao(a,n);if(!s)return null;a.index===null||a.index===void 0?r.splice(0,s.length,...s):r[a.index]=s[0]}return i===1?r[0]:r.slice(0,i)}function em(e,t){const n=e.bindings??{},i=Ri(n.position,e.position,t),r=Ri(n.rotation,e.rotation,t),a={...e.dimensions};let s=i!==null||r!==null;for(const o of Object.keys(a)){const l=Ri(n[o],a[o],t);l!==null&&(a[o]=l,s=!0)}return s?{...e,position:i??e.position,rotation:r??e.rotation,dimensions:a}:null}function tm(e,t){const{origin:n,u:i,v:r}=e.plane;return[0,1,2].map(a=>n[a]+i[a]*t[0]+r[a]*t[1])}function nm(e,t,n){const i=ro(t);return e.map(r=>{let a=!1;const s=r.vertices.map(c=>{const d=c.binding?i.get(c.binding.name):void 0;if(!d)return c;if(!Ht(d))return a=!0,{...c,free:!1};const h=ao(c.binding,n);return!h||!r.plane||h.length!==2?c:(a=!0,{...c,uv:h,world:tm(r,h)})});let o=r.edges;a&&r.kind==="profile"&&o.length===s.length&&s.length>1&&(o=s.map((c,d)=>[c.world,s[(d+1)%s.length].world]));let l=r.transform;if(r.kind!=="profile"&&r.transform){const c=em(r.transform,n);c&&(o=Ja(r.edges,r.transform,c.position,c.rotation,c.dimensions),l=c,a=!0)}return a?{...r,vertices:s,edges:o,transform:l}:r})}function Ta(e,t,n){if(!e?.length)return"fixed value in the source: the demo moves free parameters only";const i=ro(t);for(const r of e){const a=i.get(r.name),s=n?.parameters.find(o=>o.name===r.name&&o.components===r.components);if(!a||!s)return`${r.name} is not in this scene's shader, so it has nothing to move`;if(!Ht(a))return`held by ${a.drivenBy}: moving it needs the constraint solver, which runs in Python, not in this demo`}return null}function im(e,t,n,i){if(!e)return"nothing to move";if(typeof t=="number"){const r=e.vertices[t]?.binding;return Ta(r?[r]:null,n,i)}return Ta(e.transform?.bindings?.[t],n,i)}function Oa(e,t){const n={};for(const i of e??[])n[i.name]=i.index===null||i.index===void 0?t.slice(0,i.components):[t[i.index]];return n}function La(e,t=.15){const n=[1/0,1/0,1/0],i=[-1/0,-1/0,-1/0],r=s=>s.forEach((o,l)=>(n[l]=Math.min(n[l],o),i[l]=Math.max(i[l],o)));for(const s of e)s.edges.forEach(o=>o.forEach(r)),s.vertices.forEach(o=>r(o.world));if(!Number.isFinite(n[0]))return null;const a=t*Math.max(...i.map((s,o)=>s-n[o]));return{center:[0,1,2].map(s=>(n[s]+i[s])/2),half:[0,1,2].map(s=>(i[s]-n[s])/2+a),scale:1}}var rm=S("<small>"),am=S("<small class=demo-lock>constraint-driven"),sm=S("<small class=demo-lock-reason>Held by <!>. Moving it needs the constraint solver, which runs in Python, so it stays at its shipped value here."),om=S("<div class=demo-parameter><div class=demo-parameter-head><code>"),lm=S("<label class=demo-component><span></span><input type=range><input type=number>"),cm=S('<button type=button title="Back to the values the scene ships with"data-testid=parameters-reset>Reset'),dm=S("<aside class=demo-parameters data-testid=parameter-panel><div class=demo-parameter-groups>"),um=S('<button type=button class=demo-select title="Select it in the viewport">select');const hm=["u","v"],fm=["x","y","z"];function Da(e,t){return e.components===1?null:(e.components===2?hm:fm)[t]??String(t)}const Ra=e=>Math.max(0,Math.min(6,-Math.floor(Math.log10(e))));function pm(e){const t=()=>!Ht(e.parameter),n=(i,r)=>{const a=[...e.value];a[i]=no(e.parameter,i,r),e.onChange(a)};return(()=>{var i=om(),r=i.firstChild,a=r.firstChild;return i.addEventListener("mouseleave",()=>e.parameter.owner&&it(null)),i.addEventListener("mouseenter",()=>e.parameter.owner&&it({nodeId:e.parameter.owner,vertexIndex:null})),g(a,()=>e.parameter.name),g(r,b(G,{get when(){return e.parameter.unit},get children(){var s=rm();return g(s,()=>e.parameter.unit),s}}),null),g(r,b(G,{get when(){return t()},get children(){var s=am();return D(()=>P(s,"title",e.parameter.drivenBy??void 0)),s}}),null),g(i,b(_o,{get each(){return e.value},children:(s,o)=>(()=>{var l=lm(),c=l.firstChild,d=c.nextSibling,h=d.nextSibling;return g(c,()=>Da(e.parameter,o)),d.$$input=u=>n(o,Number(u.currentTarget.value)),h.addEventListener("change",u=>{const p=Number(u.currentTarget.value);Number.isFinite(p)?n(o,p):u.currentTarget.value=s().toFixed(Ra(e.parameter.step))}),D(u=>{var p=e.parameter.lower[o],f=e.parameter.upper[o],m=e.parameter.step,w=t(),x=`${e.parameter.name} ${Da(e.parameter,o)??""}`.trim(),y=`slider-${e.parameter.name}-${o}`,_=e.parameter.lower[o],$=e.parameter.upper[o],E=e.parameter.step,I=t(),A=`field-${e.parameter.name}-${o}`;return p!==u.e&&P(d,"min",u.e=p),f!==u.t&&P(d,"max",u.t=f),m!==u.a&&P(d,"step",u.a=m),w!==u.o&&(d.disabled=u.o=w),x!==u.i&&P(d,"aria-label",u.i=x),y!==u.n&&P(d,"data-testid",u.n=y),_!==u.s&&P(h,"min",u.s=_),$!==u.h&&P(h,"max",u.h=$),E!==u.r&&P(h,"step",u.r=E),I!==u.d&&(h.disabled=u.d=I),A!==u.l&&P(h,"data-testid",u.l=A),u},{e:void 0,t:void 0,a:void 0,o:void 0,i:void 0,n:void 0,s:void 0,h:void 0,r:void 0,d:void 0,l:void 0}),D(()=>d.value=s()),D(()=>h.value=s().toFixed(Ra(e.parameter.step))),l})()}),null),g(i,b(G,{get when(){return t()},get children(){var s=sm(),o=s.firstChild,l=o.nextSibling;return l.nextSibling,g(s,()=>e.parameter.drivenBy,l),s}}),null),D(s=>{var o=!!t(),l=!!e.owned,c=`parameter-${e.parameter.name}`;return o!==s.e&&i.classList.toggle("locked",s.e=o),l!==s.t&&i.classList.toggle("owned",s.t=l),c!==s.a&&P(i,"data-testid",s.a=c),s},{e:void 0,t:void 0,a:void 0}),i})()}function mm(e){const t=Q(()=>{const r=new Map(be().map(c=>[c.id,c.name??c.id])),a=new Map(be().map((c,d)=>[c.id,d])),s=c=>c.owner??`stem-${c.name.includes(".")?c.name.split(".")[0]:"scene"}`,o=new Map;for(const c of e.parameters){const d=s(c);o.set(d,[...o.get(d)??[],c])}const l=c=>c.startsWith("stem-")?-1:a.get(c)??0;return[...o.entries()].sort(([c],[d])=>l(c)-l(d)).map(([c,d])=>{const h=c.startsWith("stem-")?null:c,u=c.slice(5);return{key:c,owner:h,title:h!==null?r.get(h)??h:u==="scene"?"Scene":u,parameters:d,free:d.filter(Ht).length}})}),n=()=>e.parameters.filter(Ht).length,i=r=>r!==null&&(Ie()?.nodeId===r||nt()?.nodeId===r);return(()=>{var r=dm(),a=r.firstChild;return g(r,b(Sr,{get kicker(){return`${n()} free · ${e.parameters.length-n()} held`},title:"Parameters",get children(){var s=cm();return s.firstChild,s.$$click=()=>e.onReset(),g(s,(()=>{var o=X(()=>!!e.edited);return()=>o()?` · ${e.edited}`:""})(),null),D(()=>s.disabled=e.edited===0),s}}),a),g(a,b(Ae,{get each(){return t()},children:s=>b(Jc,{get title(){return s.title},get count(){return`${s.free}/${s.parameters.length}`},get testId(){return`parameter-group-${s.key}`},get actions(){return b(G,{get when(){return s.owner},get children(){var o=um();return o.$$click=()=>de({nodeId:s.owner,vertexIndex:null}),o}})},get children(){return b(Ae,{get each(){return s.parameters},children:o=>b(pm,{parameter:o,get value(){return e.values[o.name]??o.value},get owned(){return i(s.owner)},onChange:l=>e.onChange(o.name,l)})})}})})),r})()}Ce(["input","click"]);var gm=S("<i class=dot>"),vm=S('<div class="app demo"data-mode=model><header class="toolbar demo-toolbar"><div class=brand><span class=mark>cj</span><span>CADJOINT</span></div><span class=demo-badge title="A static page: no Python runs; everything below runs in your browser">public demo</span><div class=spacer></div><span data-testid=status></span><span class=demo-device data-testid=compute-device title="Where jax-js runs the meshing"></span><button type=button class=primary data-testid=export-stl></button></header><main class=demo-desk><section class=demo-viewport></section><div class=demo-side>'),bm=S("<a class=demo-download data-testid=stl-download>"),ym=S('<div class=view-controls data-testid=view-controls><button type=button class=viewport-bar-button title="Drag sketch points  (V)"data-testid=pick-points><span class=rail-word>Points</span></button><button type=button class=viewport-bar-button title="Select solids and drag their gizmo  (O)"data-testid=pick-objects><span class=rail-word>Objects'),wm=S("<div class=demo-nogpu data-testid=no-webgpu><b>The viewport cannot draw here: it needs WebGPU.</b><p> The model is ray-marched on the GPU; a current Chrome or Edge, or Safari 26 and later, can draw it. "),_m=S("<p class=demo-error>"),xm=S("<section class=demo-about data-testid=scene-about><h1></h1><p></p><p class=demo-note>Dual contouring in the browser; the exact B-rep, simulation and optimisation run in the <a href=https://github.com/andrinr/cadjoint#readme target=_blank rel=noreferrer>Python playground</a>.");const Ba=64,km=["64","96","128","160"],Sm=400,Bi=e=>e.replaceAll("_"," ");function $m(e,t){const n={};for(const[i,r]of Object.entries(e))r.some((a,s)=>a!==t[i]?.[s])&&(n[i]=[...r]);return n}function Pm(){let e;const t=new Promise(j=>e=j),[n,i]=T(""),r=new Up({onReady:()=>e(!0),onError:j=>{i(R=>R||j),e(!1),J({kind:"error",text:j})}}),a=op(r);ld(Wp);const[s,o]=T(null),[l,c]=T({});let d={};const[h,u]=T("pending"),[p,f]=T(null),[m,w]=T(!1),[x,y]=T("96"),[_,$]=T(null),E=()=>s()?.parameters??[],I=Q(()=>Object.keys(Qp(E(),l())).length),A=()=>{const j=s();return j?Ca(j.scene_graph,l()):null};let B=Promise.resolve(),N=0;const L=()=>{const j=s();if(!j||h()!=="ok")return Promise.resolve();const R=++N,U=l();return B=B.then(async()=>{if(R!==N)return;const v=await sp(Ca(j.scene_graph,U));R===N&&(await r.setShaders({preview:v.preview_shader,path:v.path_shader,present:v.present_shader,program:v.program}),d=U)}),B};let H;const Z=()=>{clearTimeout(H),H=setTimeout(()=>void L(),120)},ie=j=>{const R=s();R&&c(U=>Jp(R.parameters,U,j))};Te(Lr(l,j=>{const R=s();if(!R||(Fl(nm(R.construction,R.parameters,j)),h()!=="ok"))return;const U=$m(j,d);Object.keys(U).length&&r.setParameterOverrides(U),Z()}));const ce=j=>async()=>{J({kind:"",text:`${j} needs the Python playground; the demo edits parameters only`})},Re=async(j,R,U,v)=>{const C=be().find(M=>M.kind==="profile"&&M.line===R)?.vertices[U]?.binding;if(j!=="set_vertex"||!C||!v)return ce("Changing a sketch's points")();ie(Oa([C],v)),clearTimeout(H),await L()},ue=async(j,R,U,v)=>{const C=be().find(M=>M.transform?.line===j&&M.transform.call===R)?.transform?.bindings?.[U];if(!C)return ce(`Changing ${U}`)();ie(Oa(C,typeof v=="number"?[v]:v)),clearTimeout(H),await L()},ze=(j,R)=>im(ae(j),R,E(),r.parameterProgram);ft(qu({"select-object":{run:()=>yt("object"),checked:()=>It()==="object"},"select-vertex":{run:()=>yt("vertex"),checked:()=>It()==="vertex"},"gizmo-translate":{run:()=>Kn("translate")},"gizmo-scale":{run:()=>Kn("scale")}})),Te(()=>{r.gizmoMode=lc(),r.invalidate()});const Ue=async j=>{J({kind:"",text:`Loading ${Bi(j)}…`});const R=await Xp(j),U=Ma(R.parameters);d=U,uo(()=>{de(null),pi(null),rc(`${R.name}.py`),jl(R.elements),Ul(R.scene_tree),Nl(R.relations),o(R),c(U)});const v=new URL(location.href);v.searchParams.set("scene",j),history.replaceState(null,"",v);const k=dr(R.scene_graph.root)??La(R.construction);k&&(r.camera={...r.camera,target:[...k.center],distance:Math.hypot(...k.half)*2.2}),h()==="ok"&&(clearTimeout(H),await L());const C=R.parameters.filter(M=>M.drivenBy===null).length;J({kind:"ready",text:`${C} free parameter${C===1?"":"s"} · drag a ring in the viewport or move a slider`})},qe=async j=>{const R=A(),U=R&&(dr(R.root)??La(be()));if(!R||!U)throw new Error("this scene has no bounding box to mesh in");const{meshScene:v}=await tt(async()=>{const{meshScene:C}=await import("./jaxjs-Dm1kCLlx.js").then(M=>M.j);return{meshScene:C}},[],import.meta.url),k=await v(R,U,j);return f(k.device),k},Ne=()=>a.display().showMeshWireframe||a.display().showMeshEdges;let xe,et=0;Te(Lr([Ne,l,s],([j])=>{if(clearTimeout(xe),!j){pi(null);return}const R=++et;xe=setTimeout(async()=>{try{J({kind:"",text:"Meshing the overlay on jax-js…"});const U=await qe(Ba);if(R!==et||!Ne())return;const v=Zp(U.vertices,U.triangles,Ba);pi(v),J({kind:"ready",text:`Overlay: ${v.wire.length.toLocaleString()} edges, ${v.sharp.length.toLocaleString()} creases; ${U.cells.join("×")} cells on ${U.device} in ${(U.sampleMs+U.meshMs).toFixed(0)} ms`})}catch(U){J({kind:"error",text:`Wireframe failed: ${U instanceof Error?U.message:U}`})}},Sm)}));const Wt=async()=>{const j=s();if(j){w(!0),J({kind:"",text:`Meshing ${Bi(j.name)} on jax-js…`});try{const R=await qe(Number(x())),{stlBlob:U}=await tt(async()=>{const{stlBlob:W}=await import("./jaxjs-Dm1kCLlx.js").then(F=>F.j);return{stlBlob:W}},[],import.meta.url),v=_();v&&URL.revokeObjectURL(v.url);const k=`${j.name}${I()?"-edited":""}.stl`,C=URL.createObjectURL(U(R));$({url:C,name:k}),Object.assign(document.createElement("a"),{href:C,download:k}).click(),J({kind:"ready",text:`${k}: ${(R.triangles.length/3).toLocaleString()} triangles, ${R.cells.join("×")} cells on ${R.device}, ${(R.sampleMs+R.meshMs).toFixed(0)} ms`+(R.badEdges?`, ${R.badEdges} open edges`:"")})}catch(R){J({kind:"error",text:`Export failed: ${R instanceof Error?R.message:R}`})}finally{w(!1)}}};pr(async()=>{yt("vertex");const j=new URLSearchParams(location.search).get("scene"),R=j&&_n.includes(j)?j:_n.includes("bracket")?"bracket":_n[0];t.then(U=>{u(U?"ok":"missing"),U&&L()}),tt(async()=>{const{computeDevice:U}=await import("./jaxjs-Dm1kCLlx.js").then(v=>v.j);return{computeDevice:U}},[],import.meta.url).then(({computeDevice:U})=>U()).then(f,()=>f("none"));try{await Ue(R)}catch(U){J({kind:"error",text:`Could not load ${R}: ${U instanceof Error?U.message:U}`})}}),ft(()=>{clearTimeout(H),clearTimeout(xe)}),window.__cadjointDemo=()=>({scene:s()?.name??null,values:l(),uploads:r.shaderStats.parameterUploads,pipelineBuilds:r.shaderStats.pipelineBuilds,device:p()});const st=()=>({display:a.display(),presets:a.presets(),selectedPreset:a.selectedPreset(),pathTracing:a.pathTracing(),quality:a.quality(),onChange:a.applyDisplay,onQualityChange:a.applyQuality,onPresetActivate:a.activateRenderPreset,onPresetSave:a.saveRenderPreset,onPresetReset:a.resetRenderPreset,onPathTracingChange:a.applyPathTracing});return(()=>{var j=vm(),R=j.firstChild,U=R.firstChild,v=U.nextSibling,k=v.nextSibling,C=k.nextSibling,M=C.nextSibling,W=M.nextSibling,F=R.nextSibling,O=F.firstChild,Y=O.nextSibling;return g(R,b(Kr,{get value(){return s()?.name??""},get options(){return _n.map(z=>({value:z,label:Bi(z)}))},onChange:z=>void Ue(z),"aria-label":"Scene",testId:"scene-picker"}),k),g(R,b(tr,{kind:"brep",testId:"demo-tier"}),k),g(C,b(G,{get when(){return mi().text},get children(){return[gm(),X(()=>mi().text)]}})),g(M,(()=>{var z=X(()=>p()===null);return()=>z()?"jax-js …":X(()=>p()==="none")()?"no jax-js device":`jax-js · ${p()}`})()),g(R,b(Kr,{get value(){return x()},get options(){return km.map(z=>({value:z,label:`${z} cells`}))},onChange:y,"aria-label":"Export resolution",title:"Cells on the longest side of the export lattice",testId:"export-cells"}),W),W.$$click=()=>void Wt(),g(W,()=>m()?"Meshing…":"Export STL"),g(R,b(G,{get when(){return _()},children:z=>(()=>{var te=bm();return g(te,()=>z().name),D(ge=>{var Ge=z().url,pt=z().name;return Ge!==ge.e&&P(te,"href",ge.e=Ge),pt!==ge.t&&P(te,"download",ge.t=pt),ge},{e:void 0,t:void 0}),te})()}),null),g(F,b(xd,{}),O),g(O,b(jh,{renderer:r,get display(){return a.display()},onPatch:Re,onLiveParameters:ie,lockedHandle:ze,onSetValue:ue,get onAddPrimitive(){return ce("Adding a solid")},get onAddSketch(){return ce("Adding a sketch")},get onSketchOnFace(){return ce("Sketching on a face")},get onAddConstraint(){return ce("Adding a constraint")},get onAddLoft(){return ce("Lofting")},get onDeleteObject(){return ce("Deleting an object")},get onAssignMaterial(){return ce("Assigning a material")},get overlay(){return[(()=>{var z=ym(),te=z.firstChild,ge=te.firstChild,Ge=te.nextSibling,pt=Ge.firstChild;return te.$$click=()=>yt("vertex"),g(te,b(zc,{size:16}),ge),Ge.$$click=()=>yt("object"),g(Ge,b(er,{size:16}),pt),g(z,b(su,{get shown(){return a.display().showOverlays},onToggle:We=>a.applyDisplay({showOverlays:We})}),null),g(z,b(ou,{get projection(){return a.display().projection},onProjection:We=>a.applyDisplay({projection:We})}),null),g(z,b(xf,{get render(){return st()}}),null),D(We=>{var Tr=It()==="vertex",Or=It()==="object";return Tr!==We.e&&te.classList.toggle("active",We.e=Tr),Or!==We.t&&Ge.classList.toggle("active",We.t=Or),We},{e:void 0,t:void 0}),z})(),b(au,{get yaw(){return Ki().yaw},get pitch(){return Ki().pitch},get projection(){return a.display().projection},get onPreset(){return a.applyPreset},onProjection:z=>a.applyDisplay({projection:z}),onOrbit:(z,te)=>{r.camera={...r.camera,yaw:z,pitch:te},xr({yaw:z,pitch:te}),r.invalidate()}}),b(G,{get when(){return h()==="missing"},get children(){var z=wm(),te=z.firstChild,ge=te.nextSibling,Ge=ge.firstChild;return g(ge,n,Ge),g(ge,(()=>{var pt=X(()=>p()==="wasm");return()=>pt()?"The parameters and the STL export still work: meshing runs on WebAssembly.":p()==="none"?"This browser cannot run jax-js on WebAssembly either, so the STL export is off too.":""})(),null),z}}),b(G,{get when(){return dn()},get children(){var z=_m();return g(z,dn),z}})]}})),g(Y,b(G,{get when(){return s()},children:z=>(()=>{var te=xm(),ge=te.firstChild,Ge=ge.nextSibling,pt=Ge.nextSibling,We=pt.firstChild;return We.nextSibling,g(ge,()=>z().title),g(Ge,()=>z().description),te})()}),null),g(Y,b(mm,{get parameters(){return E()},get values(){return l()},get edited(){return I()},onChange:(z,te)=>ie({[z]:te}),onReset:()=>{const z=s();z&&c(Ma(z.parameters))}}),null),D(z=>{var te=`status ${mi().kind}`,ge=m()||!s()||p()==="none";return te!==z.e&&Lt(C,z.e=te),ge!==z.t&&(W.disabled=z.t=ge),z},{e:void 0,t:void 0}),j})()}Ce(["click"]);const so=document.getElementById("root");if(!so)throw new Error("Missing #root element");To(()=>b(Pm,{}),so);export{tt as _};
