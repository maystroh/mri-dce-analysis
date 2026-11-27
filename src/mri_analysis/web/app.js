"use strict";
const $ = id => document.getElementById(id);
const planes = {sagittal:{axis:0,left:"P",right:"A",top:"S",bottom:"I"},coronal:{axis:1,left:"L",right:"R",top:"S",bottom:"I"},axial:{axis:2,left:"L",right:"R",top:"A",bottom:"P"}};
let overview, run, selected, details, position, revision=0, filtered=[];
const make = (tag, text, cls) => {const el=document.createElement(tag); if(text!==undefined)el.textContent=text; if(cls)el.className=cls; return el;};
const number = (value,digits=2) => value!==undefined&&value!==null&&value!==""&&Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "unavailable";
async function json(url){const res=await fetch(url);if(!res.ok)throw new Error(await res.text());return res.json();}
const params = extra => new URLSearchParams({run:run.name,case:selected,...extra});
function status(text){$("status").textContent=text;}
function reviewStore(){try{return JSON.parse(localStorage.getItem("mri-review:"+overview.review_key)||"{}");}catch{return {};}}
function reviewKey(){return JSON.stringify([run.name,selected]);}
const MASKS=["region","nipple"],FIELDS=["overall","region","nipple"],DECISIONS=["Accurate","Incorrect","Needs review"];
let savedReviews={},reviewWritable=false,reviewFile="",dirty=false,saving=false;
function choice(field){return document.querySelector('input[name="'+field+'"]:checked')?.value||"";}
function setChoice(field,value){for(const input of document.querySelectorAll('input[name="'+field+'"]'))input.checked=input.value===(value||"");}
function currentReview(){return {overall:choice("overall"),region:choice("region"),nipple:choice("nipple"),note:$("note").value.trim()};}
function effective(review){const result={};for(const mask of MASKS)result[mask]=review[mask]||review.overall||"";return result;}
function hasReview(review){return !!(review&&(review.overall||review.region||review.nipple));}
function localReview(runName,caseId){
  const local=reviewStore()[JSON.stringify([runName,caseId])];
  return local?{overall:local.overall??local.decision??"",region:local.region||"",nipple:local.nipple||"",note:local.note||"",updated_at:local.updated_at}:null;
}
// Server file first; this browser's copy covers read-only folders and older decision-only reviews.
function reviewFor(caseId){return savedReviews[caseId]||localReview(run.name,caseId);}
function reviewClass(review){
  if(!hasReview(review))return "";
  const values=Object.values(effective(review));
  if(values.includes("Incorrect"))return " rev-bad";
  if(values.includes("Needs review")||values.includes(""))return " rev-check";
  return " rev-ok";
}
function reviewLabel(review){
  const e=effective(review);
  return e.region===e.nipple?e.region:"region "+(e.region||"—")+" · nipple "+(e.nipple||"—");
}
function reviewedCount(){return run.cases.filter(c=>hasReview(reviewFor(c.case))).length;}
function showProgress(){const done=reviewedCount();$("review-progress").textContent="Reviewed "+done+" / "+run.cases.length;$("progress-bar").style.width=(run.cases.length?100*done/run.cases.length:0)+"%";}
function markDirty(){dirty=true;const e=effective(currentReview());$("same-region").textContent="Same"+(choice("overall")?" ("+choice("overall")+")":"");$("same-nipple").textContent="Same"+(choice("overall")?" ("+choice("overall")+")":"");$("review-saved").textContent="Unsaved · region: "+(e.region||"not set")+" · nipple: "+(e.nipple||"not set");}
function fillReview(review){
  review=review||{};
  for(const field of FIELDS)setChoice(field,review[field]);
  $("note").value=review.note||"";
  markDirty();dirty=false;
  $("review-saved").textContent=hasReview(review)?"Reviewed "+(review.updated_at?new Date(review.updated_at).toLocaleString():"")+" · "+reviewLabel(review):(reviewWritable?"Not reviewed yet. Pick All masks, change a mask only if it differs, then Save & next.":"Not reviewed yet. The results folder is read-only, so reviews stay in this browser.");
}
function saveLocal(runName,caseId,review){try{const store=reviewStore();store[JSON.stringify([runName,caseId])]={run:runName,case:caseId,...review,updated_at:new Date().toISOString()};localStorage.setItem("mri-review:"+overview.review_key,JSON.stringify(store));return true;}catch{return false;}}
async function postReview(runName,caseId,review){
  const res=await fetch("/api/review",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({run:runName,case:caseId,...review})});
  const body=await res.json().catch(()=>({}));
  if(!res.ok)throw new Error(body.error||res.statusText);
  return body.review;
}
async function storeReview(review){
  const runName=run.name,caseId=selected;
  const local=saveLocal(runName,caseId,review);
  let saved={...review,effective:effective(review),updated_at:new Date().toISOString()};
  if(reviewWritable)saved=await postReview(runName,caseId,review);
  else if(!local)throw new Error("the results folder is read-only and browser storage is unavailable");
  if(hasReview(saved))savedReviews[caseId]=saved;else delete savedReviews[caseId];
  dirty=false;caseList();showProgress();
}
function nextCase(){
  const index=filtered.findIndex(c=>c.case===selected);
  const order=[...filtered.slice(index+1),...filtered.slice(0,Math.max(index,0))];
  return order.find(c=>!hasReview(reviewFor(c.case)))||null;
}
async function saveAndNext(){
  if(!selected||saving)return;
  const review=currentReview(),e=effective(review);
  if(!e.region||!e.nipple){$("review-saved").textContent="Choose All masks, or a decision for both Breast region and Nipple, before saving.";return;}
  saving=true;$("save-next").disabled=true;$("review-saved").textContent="Saving…";
  try{
    await storeReview(review);
    const next=nextCase();
    if(next)selectCase(next.case,true);
    else{fillReview(reviewFor(selected));$("review-saved").textContent="Saved. Every case in this list is reviewed.";status("All "+filtered.length+" listed cases are reviewed. Use Export reviews to download them.");}
  }catch(error){$("review-saved").textContent="Not saved: "+error.message;}
  finally{saving=false;$("save-next").disabled=false;}
}
async function clearReview(){
  if(!selected||!hasReview(reviewFor(selected))&&!dirty){fillReview(null);return;}
  if(!confirm("Remove the saved review of "+selected+"?"))return;
  try{await storeReview({overall:"",region:"",nipple:"",note:""});fillReview(null);}catch(error){$("review-saved").textContent="Not cleared: "+error.message;}
}
async function loadReviews(name){try{const data=await json("/api/reviews?"+new URLSearchParams({run:name}));savedReviews=data.reviews||{};reviewWritable=!!data.writable;reviewFile=data.file;}catch{savedReviews={};reviewWritable=false;reviewFile="";}}
function reviewEnabled(enabled){for(const el of document.querySelectorAll("#review-panel input,#review-panel button"))el.disabled=!enabled;}
function caseList(){filtered=run.cases.filter(c=>c.case.toLowerCase().includes($("search").value.toLowerCase())&&(!$("review-only").checked||c.needs_review)&&(!$("unreviewed-only").checked||c.case===selected||!hasReview(reviewFor(c.case))));$("count").textContent=filtered.length+" / "+run.cases.length;$("cases").replaceChildren();for(const c of filtered){const review=reviewFor(c.case);const button=make("button",c.case,"case-button"+(c.case===selected?" selected":"")+reviewClass(review));button.dataset.case=c.case;const state=[c.status!=="ok"?c.status:"",c.needs_review?"QC review":""].filter(Boolean).join(" · ");if(state)button.append(make("small",state,c.needs_review?"flag":""));if(hasReview(review))button.append(make("small","✓ "+reviewLabel(review),"reviewed"));button.onclick=()=>selectCase(c.case);$("cases").append(button);}}
function stat(label,value){const el=make("div",undefined,"stat");el.append(make("span",label),make("strong",value));return el;}
function motionTable(series){
  const table=make("table"),head=make("tr"),thead=make("thead"),body=make("tbody");
  for(const label of ["Phase / file","Mean displacement (mm)","95th percentile (mm)","Rotation (°)","Registration stop reason"])head.append(make("th",label));
  thead.append(head);table.append(thead);
  for(const pair of series.pairs||[]){
    const row=make("tr");
    row.append(make("td",pair.phase+" · "+(pair.moving_file||"unavailable")),make("td",number(pair.mean_mm)),make("td",number(pair.p95_mm)),make("td",number(pair.rotation_deg)),make("td",pair.stop_condition||"unavailable"));
    body.append(row);
  }
  for(const phase of series.missing_phases||[]){
    const row=make("tr"),cell=make("td","Phase "+phase+": unavailable / registration failed");
    cell.colSpan=5;row.append(cell);body.append(row);
  }
  table.append(body);const scroll=make("div",undefined,"motion-table");scroll.append(table);return scroll;
}
function renderMotion(motion){
  const panel=$("motion-panel"), target=$("motion-results");target.replaceChildren();
  const data=motion||{status:"not_requested",reason:"Motion analysis was not requested"};
  panel.hidden=false;if(data.status==="not_requested"){$("motion-status").textContent="Motion was not run for this analysis. Add --motion to the run command to measure it.";return;}
  $("motion-status").textContent=data.status+(data.score_mm!==undefined&&data.score_mm!==null?" · worst phase displacement "+number(data.score_mm)+" mm":"")+(data.reason?" · "+data.reason:"");
  target.append(make("p","Each phase is compared with the reference image. These graphs show movement relative to that reference, rather than changes between consecutive phases.","hint"));
  if(data.reference_phase)target.append(make("p","Reference: "+data.reference_phase+(data.reference_substituted?" (first available phase; no pre-contrast reference)":""),"hint motion-reference"));
  for(const warning of data.warnings||[])target.append(make("p",warning,"warnings"));
  for(const [name,series] of Object.entries(data.series||{})){
    const block=make("div",undefined,"motion-series");
    block.append(make("h3",name.charAt(0).toUpperCase()+name.slice(1)+" · "+series.status+" · ROI "+number(series.roi_cm3,1)+" cm³"));
    motionCharts.render(block,series,Object.values(data.series));
    const tableOption=make("details",undefined,"motion-details");
    tableOption.append(make("summary","Show detailed measurements (table)"));
    let tableLoaded=false;
    tableOption.ontoggle=()=>{if(tableOption.open&&!tableLoaded){tableOption.append(motionTable(series));tableLoaded=true;}};
    block.append(tableOption);if(series.note)block.append(make("p",series.note,"warnings"));target.append(block);
  }
  if(Object.keys(data.series||{}).length)target.append(make("p","Rigid motion measurements; larger values mean more movement. Inspect the registration evidence. No pass/fail threshold is applied. The MRI panes show the reference image.","hint"));
}
async function selectRun(name){revision++;run=overview.runs.find(r=>r.name===name);details=null;selected=null;$("cases").replaceChildren();await loadReviews(name);if(run.name!==name)return;dirty=false;showProgress();$("summary").replaceChildren(stat(run.summary.exams===undefined?"Volumes":"Exams",run.summary.exams??run.cases.length),stat("Completed",run.summary.exams_ok??"—"),stat("Failed",run.summary.exams_failed??"—"),stat("Missing nipples",run.summary.nipple_missing_sides??"—"));if(run.motion?.exams!==undefined)$("summary").append(stat("Motion measured",run.motion.exams_ok),stat("Motion unavailable",run.motion.exams_skipped),stat("Motion incomplete",run.motion.exams_failed));if(run.annotations.scored!==undefined)$("summary").append(stat("Region Dice · median",number(run.annotations.breast_region_dice?.median,3)),stat("Nipple error · median mm",number(run.annotations.nipple_centroid_distance_mm?.median)));renderRunQC();caseList();if(filtered.length)selectCase(filtered[0].case);else{$("case-title").textContent="No cases in this run";$("planes").replaceChildren();status("This run has no saved exams.");}}
async function selectCase(caseId,afterSave){if(!afterSave&&dirty&&caseId!==selected&&!confirm("Leave "+selected+" without saving its review?"))return;dirty=false;selected=caseId;details=null;fillReview(reviewFor(caseId));reviewEnabled(true);const generation=++revision;caseList();$("case-title").textContent=caseId;$("planes").replaceChildren();$("metrics").replaceChildren();$("warnings").textContent="";$("motion-results").replaceChildren();$("motion-status").textContent="Loading…";status("Loading "+caseId+"…");try{const value=await json("/api/case?"+params({}));if(generation!==revision)return;details=value;renderMotion(value.motion);$("case-status").textContent=value.status==="ok"?"":value.status;renderCaseQC(value);const metrics=[["Region Dice",value.scores.breast_region_dice],["Nipple Dice",value.scores.nipple_dice],["Selected nipple error (mm)",value.scores.nipple_centroid_distance_mm],["Nearest component error (mm)",value.scores.nipple_nearest_component_distance_mm]];for(const [label,val]of metrics){if(val!==undefined){const el=make("span",label);el.append(make("strong",number(val, label.includes("Dice")?3:2)));$("metrics").append(el);}}if(!value.available){$("warnings").textContent=value.error;status("Case evidence is available; image could not be opened.");disableJumps();return;}position=[...value.position];$("low").value=value.window[0].toFixed(2);$("high").value=value.window[1].toFixed(2);$("warnings").textContent=value.warnings.join(" · ");for(const [name,p]of Object.entries(planes)){const panel=make("div",undefined,"plane");panel.id="plane-"+name;panel.hidden=name!==$("view").value;panel.append(make("h3",name+" · RAS voxel slice"));const orient=make("div",undefined,"orientation");orient.append(make("span",p.top+" ↑"),make("span",p.left+" ←   → "+p.right));panel.append(orient);const img=make("img");img.id="image-"+name;img.alt=name+" MRI and mask overlays";img.onerror=()=>{if(generation===revision)status("A slice could not be loaded. Check the window values or the case warnings.");};img.onclick=event=>{const rect=img.getBoundingClientRect();const axes=[0,1,2].filter(i=>i!==p.axis);position[axes[0]]=Math.max(0,Math.min(value.shape[axes[0]]-1,Math.floor((event.clientX-rect.left)/rect.width*value.shape[axes[0]])));position[axes[1]]=Math.max(0,Math.min(value.shape[axes[1]]-1,value.shape[axes[1]]-1-Math.floor((event.clientY-rect.top)/rect.height*value.shape[axes[1]])));refreshSlices();};const frame=make("div",undefined,"image-frame");frame.append(img);panel.append(frame);const controls=make("div",undefined,"slice-control");for(const type of ["range","number"]){const input=make("input");input.type=type;input.min=0;input.max=value.shape[p.axis]-1;input.value=position[p.axis];input.id=type+"-"+name;input.setAttribute("aria-label",name+" slice "+type);input.oninput=()=>{position[p.axis]=Math.max(0,Math.min(value.shape[p.axis]-1,Math.round(Number(input.value)||0)));refreshSlices();};controls.append(input);}panel.append(controls);$("planes").append(panel);}disableJumps();refreshSlices();status(run.name+" · "+run.cases.length+" cases · "+(run.portable?"portable bundle":"original dataset references"));}catch(error){if(generation===revision)status("Could not load case: "+error.message);}}
function disableJumps(){for(const kind of ["gt","pred","error"]){$("jump-"+kind).disabled=!details?.available||(kind==="error"?!Object.keys(details.error_slices).length:!details.centres[kind==="gt"?"ground_truth":"prediction"].length);}}
function refreshSlices(){if(!details?.available)return;if(!Number.isFinite(Number($("low").value))||!Number.isFinite(Number($("high").value))||Number($("high").value)<=Number($("low").value)){status("Set a finite window high value greater than the low value.");return;}for(const [name,p]of Object.entries(planes)){if(name!==$("view").value)continue;$("range-"+name).value=position[p.axis];$("number-"+name).value=position[p.axis];$("image-"+name).src="/api/slice?"+params({plane:name,index:position[p.axis],mode:$("mode").value,layer:$("layer").value,opacity:$("opacity").value,low:$("low").value,high:$("high").value});}if($("mode").value==="errors"&&!details.layers.some(l=>l.startsWith("gt_")))status("Error maps require annotations. Choose Contours or Prediction for this case.");}
// One view at a time; only the shown view requests slices.
$("view").onchange=()=>{
  for(const name of Object.keys(planes)){const panel=$("plane-"+name);if(!panel)continue;panel.hidden=name!==$("view").value;if(panel.hidden)$("image-"+name).removeAttribute("src");}
  refreshSlices();
};
async function previewArtifact(file){const target=$("artifact-preview");target.replaceChildren(make("h3",file.name));try{if(/\.(png|jpg|svg)$/i.test(file.name)){const img=make("img");img.src=file.url;img.alt=file.name;target.append(img);}else if(file.name.endsWith(".csv")){const data=await json(file.url+"?table=1");const table=make("table");const head=make("thead"),row=make("tr"),keys=Object.keys(data.rows[0]||{});for(const key of keys)row.append(make("th",key));head.append(row);table.append(head);const body=make("tbody");for(const record of data.rows.slice(0,1000)){const tr=make("tr");for(const key of keys)tr.append(make("td",record[key]));body.append(tr);}table.append(body);target.append(make("p",data.rows.length+" rows · displaying up to 1,000", "hint"),table);}else if(file.name.endsWith(".pdf")){const a=make("a","Open PDF");a.href=file.url;a.target="_blank";target.append(a);}else{const res=await fetch(file.url);target.append(make("pre",await res.text()));}}catch(error){target.append(make("p",error.message));}}
function artifacts(){for(const file of overview.artifacts){const div=make("div",undefined,"artifact"),button=make("button",file.name),download=make("a","Download");button.onclick=()=>previewArtifact(file);download.href=file.url+"?download=1";div.append(button,download);$("artifacts").append(div);}}
$("run").onchange=()=>selectRun($("run").value);$("search").oninput=caseList;$("review-only").onchange=caseList;for(const id of ["mode","layer","opacity","low","high"])$(id).oninput=refreshSlices;$("unreviewed-only").onchange=caseList;for(const input of document.querySelectorAll("#review-panel input"))input.addEventListener(input.type==="radio"?"change":"input",markDirty);$("save-next").onclick=saveAndNext;for(const [id,step] of [["prev-case",-1],["next-case",1]])$(id).onclick=()=>{const index=filtered.findIndex(c=>c.case===selected),target=filtered[index+step];if(target)selectCase(target.case,true);else status(step>0?"This is the last case in the list.":"This is the first case in the list.");};$("clear-review").onclick=clearReview;window.addEventListener("beforeunload",event=>{if(dirty){event.preventDefault();event.returnValue="";}});
$("jump-gt").onclick=()=>{position=[...details.centres.ground_truth[0]];refreshSlices();};$("jump-pred").onclick=()=>{position=[...details.centres.prediction[0]];refreshSlices();};$("jump-error").onclick=()=>{position=[...(details.error_slices[$("layer").value==="nipple"?"nipple":"region"]||Object.values(details.error_slices)[0])];refreshSlices();};
$("export-reviews").onclick=()=>$("export").click();$("export").onclick=async()=>{
  const reviews=[];
  for(const r of overview.runs){
    let saved=r.name===run.name?savedReviews:{};
    if(r.name!==run.name){try{saved=(await json("/api/reviews?"+new URLSearchParams({run:r.name}))).reviews||{};}catch{}}
    for(const c of r.cases){
      const review=saved[c.case]||localReview(r.name,c.case);
      if(!hasReview(review))continue;
      const e=effective(review);
      reviews.push({run:r.name,case:c.case,breast_region:e.region,nipple:e.nipple,note:review.note||"",reviewed_at:review.updated_at||""});
    }
  }
  if(!reviews.length){status("No reviews to export yet. Save a review first.");return;}
  // One zip with every run's saved review files (CSV and JSON) and mask_reviews.json.
  const summary={format:"mri-mask-review",version:1,exported_at:new Date().toISOString(),decisions:DECISIONS,reviews};
  try{
    const res=await fetch("/api/reviews/export",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mask_reviews:summary})});
    if(!res.ok)throw new Error((await res.json().catch(()=>({}))).error||res.statusText);
    const name=(res.headers.get("Content-Disposition")||"").match(/filename="?([^";]+)"?/)?.[1]||"reviews.zip";
    const url=URL.createObjectURL(await res.blob()),a=make("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    status("Exported "+reviews.length+" reviews to "+name+".");
  }catch(error){status("Not exported: "+error.message);}
};
document.addEventListener("keydown",event=>{
  const typing=event.target.tagName==="TEXTAREA"||event.target.tagName==="SELECT"||event.target.tagName==="INPUT"&&event.target.type!=="radio";
  if(event.key==="Enter"&&!event.target.closest("button")&&(!typing||event.target.id==="note")){event.preventDefault();saveAndNext();return;}
  if(typing||event.ctrlKey||event.metaKey||event.altKey)return;
  const index=filtered.findIndex(c=>c.case===selected),key=event.key.toLowerCase();
  if(key==="j"&&index<filtered.length-1)selectCase(filtered[index+1].case);
  if(key==="k"&&index>0)selectCase(filtered[index-1].case);
  if(["1","2","3"].includes(key)&&selected){setChoice("overall",DECISIONS[Number(key)-1]);markDirty();}
});
json("/api/run").then(data=>{overview=data;for(const r of data.runs){const option=make("option",r.name);option.value=r.name;$("run").append(option);}$("run").value=data.initial_run;artifacts();selectRun(data.initial_run);}).catch(error=>status("Could not open run: "+error.message));

// ---- Readable quality control (no raw JSON on the page) ----
const yes = v => String(v).toLowerCase()==="true"||v===true;
const filled = v => v!==undefined&&v!==null&&String(v).trim()!==""&&String(v)!=="nan";
function kvTable(rows){
  const table=make("table",undefined,"kv");
  for(const [label,value,bad] of rows){if(!filled(value))continue;const tr=make("tr");tr.append(make("th",label));const td=make("td",String(value));if(bad)td.className="bad";tr.append(td);table.append(tr);}
  return table.children.length?table:make("p","No values recorded.","hint");
}
function counts(obj){return obj&&typeof obj==="object"?Object.entries(obj).map(([k,v])=>k+": "+v).join(" · "):obj;}
function block(title,content){const div=make("div",undefined,"qc-block");div.append(make("h3",title),content);return div;}
function renderCaseQC(value){
  const m=value.metrics||{},target=$("case-qc");target.replaceChildren();
  const errors=Object.entries(m).filter(([k,v])=>k.endsWith("_error")&&filled(v)).map(([k,v])=>k.replace(/_error$/,"").replace(/_/g," ")+": "+v).join(" · ");
  target.append(block("Exam",kvTable([
    ["Status",value.status,value.status!=="ok"&&value.status!=="available"],
    ["Flagged for QC review",value.needs_review?"Yes":"No",value.needs_review],
    ["Errors",errors,true],
    ["Phases",m.n_phases],["Plane",m.plane],
    ["Breast coverage",filled(m.coverage_scope)?m.coverage_scope+(yes(m.coverage_certain)?" (certain)":" (uncertain)"):"",filled(m.coverage_certain)&&!yes(m.coverage_certain)],
    ["Coverage reason",m.coverage_reason],
    ["Coverage agrees with stage 1",filled(m.coverage_agrees)?(yes(m.coverage_agrees)?"Yes":"No"):"",filled(m.coverage_agrees)&&!yes(m.coverage_agrees)],
    ["Lesion pipeline scope",filled(m.pipeline_scope)?m.pipeline_scope+(filled(m.pipeline_reason)?" — "+m.pipeline_reason:""):""],
    ["Laterality difference",m.laterality_difference_type,filled(m.laterality_difference_type)&&m.laterality_difference_type!=="none"],
    ["Left-right field of view (mm)",m.fov_lr_mm],["Smallest breast side (mm)",m.min_side_mm],
    ["Oblique angle (°)",m.oblique_deg],["Midline cut offset (mm)",m.midline_cut_offset_mm],
    ["Breasts reported",m.sides_reported??m.n_sides_reported],
    ["Breasts needing review",m.sides_needing_review,Number(m.sides_needing_review)>0],
    ["Breasts without nipple",m.sides_without_nipple,Number(m.sides_without_nipple)>0],
    ["Model",filled(m.model_name)?m.model_name+" v"+m.model_version:""],
    ["Prediction reused from cache",filled(m.reused)?(yes(m.reused)?"Yes":"No"):""],
    ["DICOM source",value.dicom_source],
  ])));
  const sides=value.sides||[];
  if(sides.length){
    const cols=[["Breast",s=>s.side],["Needs review",s=>yes(s.needs_review)?"Yes":"No",s=>yes(s.needs_review)],
      ["Fit to use",s=>yes(s.fit_to_consume)?"Yes":"No",s=>!yes(s.fit_to_consume)],["Unfit reason",s=>s.unfit_reason,s=>filled(s.unfit_reason)],
      ["Region volume (cm³)",s=>filled(s.region_volume_mm3)?number(s.region_volume_mm3/1000,1):""],
      ["Region plausibility",s=>s.region_plausibility,s=>filled(s.region_plausibility)&&s.region_plausibility!=="plausible"],
      ["Region parts",s=>s.region_components,s=>Number(s.region_components)>1],
      ["Touches left-right edge",s=>yes(s.region_touches_lr_border)?"Yes":"No",s=>yes(s.region_touches_lr_border)],
      ["Nipples found",s=>s.n_nipples,s=>Number(s.n_nipples)!==1],
      ["Selected nipple volume (mm³)",s=>s.selected_nipple_volume_mm3],
      ["Selected nipple position (mm)",s=>[s.selected_nipple_x_mm,s.selected_nipple_y_mm,s.selected_nipple_z_mm].every(filled)?[s.selected_nipple_x_mm,s.selected_nipple_y_mm,s.selected_nipple_z_mm].join(", "):""]];
    const table=make("table",undefined,"qc-table"),head=make("tr");
    for(const [label]of cols)head.append(make("th",label));
    const thead=make("thead");thead.append(head);table.append(thead);const body=make("tbody");
    for(const s of sides){const tr=make("tr");for(const [,get,bad]of cols){const td=make("td",filled(get(s))?String(get(s)):"—");if(bad&&bad(s))td.className="bad";tr.append(td);}body.append(tr);}
    table.append(body);const scroll=make("div",undefined,"motion-table");scroll.append(table);
    target.append(block("Per breast",scroll));
  }
  const scores=Object.entries(value.scores||{}).filter(([k,v])=>k!=="case"&&filled(v)&&!String(v).startsWith("["));
  if(scores.length)target.append(block("Annotation scores",kvTable(scores.map(([k,v])=>[k.replace(/_/g," "),Number.isFinite(Number(v))?number(v,3):v]))));
  const meta=Object.entries(value.dataset_metadata||{}).filter(([,v])=>filled(v)&&typeof v!=="object");
  if(meta.length)target.append(block("Dataset metadata",kvTable(meta.map(([k,v])=>[k.replace(/_/g," "),v]))));
  $("case-download").href="/api/case/export?"+params({});
  $("case-download").textContent="Case QC and measurements · "+selected+" (JSON)";
}
function renderRunQC(){
  const s=run.summary||{},g=run.geometry||{},a=run.annotations||{},mo=run.motion||{},target=$("run-qc");target.replaceChildren();
  target.append(block("Landmarks",kvTable([
    ["Model",s.model?s.model.name+" v"+s.model.version:""],["Exams",s.exams],["Exams completed",s.exams_ok],
    ["Exams failed",s.exams_failed,Number(s.exams_failed)>0],["Breasts",s.sides],
    ["Breasts flagged for review",s.review_rows,Number(s.review_rows)>0],["Breasts without nipple",s.nipple_missing_sides,Number(s.nipple_missing_sides)>0],
    ["Region plausibility",counts(s.region_plausibility)],["Fit to use",counts(s.fit_to_consume)],
    ["Breast coverage",counts(s.coverage_scope)],["Coverage certain",counts(s.coverage_certain)],
    ["Coverage differs from stage 1",s.coverage_disagrees_with_stage1,Number(s.coverage_disagrees_with_stage1)>0],
    ["Plane",counts(s.plane)],["Exam status",counts(s.exam_status)],
  ])));
  if(Object.keys(g).length)target.append(block("Image conversion and geometry",kvTable([
    ["Studies converted",g.conversion?g.conversion.converted+" of "+g.conversion.studies:""],["Conversion failures",g.conversion?.failed,Number(g.conversion?.failed)>0],
    ["Volumes",g.volumes],["Trusted volumes",g.trusted],["Uncertain volumes",g.uncertain,Number(g.uncertain)>0],
    ["View",counts(g.view)],["Laterality",counts(g.laterality)],["Phase labels",counts(g.label)],["Flags",Object.keys(g.flags||{}).length?counts(g.flags):""],
  ])));
  if(a.scored!==undefined)target.append(block("Annotation agreement",kvTable([
    ["Scored exams",a.scored],["Region Dice · median",number(a.breast_region_dice?.median,3)],["Region Dice · mean",number(a.breast_region_dice?.mean,3)],
    ["Nipple error · median (mm)",number(a.nipple_centroid_distance_mm?.median)],["Nipple error · mean (mm)",number(a.nipple_centroid_distance_mm?.mean)],
  ])));
  target.append(block("Motion",mo.exams!==undefined?kvTable([["Exams",mo.exams],["Measured",mo.exams_ok],["Skipped",mo.exams_skipped,Number(mo.exams_skipped)>0],["Incomplete",mo.exams_failed,Number(mo.exams_failed)>0],["Measurement",mo.measurement]]):make("p","Motion was not run for this analysis. Add --motion to the run command to measure it.","hint")));
  $("run-download").href="/api/run/export?"+new URLSearchParams({run:run.name});
  $("run-download").textContent="Run summary · "+run.name+" (JSON)";
  for(const kind of ["csv","json"]){const link=$("reviews-"+kind);link.href="/api/reviews/download?"+new URLSearchParams({run:run.name,format:kind});link.textContent="Saved reviews · "+run.name+" ("+kind.toUpperCase()+")";}
}
