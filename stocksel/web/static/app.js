/* Stocksel research workspace. No remote scripts, analytics, or order execution. */
'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const number = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
const num = (value, digits = 2) => number(value) ? Number(value).toLocaleString(undefined, {maximumFractionDigits:digits, minimumFractionDigits:digits}) : '—';
const percent = (value, digits = 1) => number(value) ? `${num(Number(value) * 100, digits)}%` : '—';
const signedPercent = value => number(value) ? `${Number(value) > 0 ? '+' : ''}${percent(value)}` : '—';
const amount = value => number(value) ? `${Number(value) < 0 ? '−' : ''}$${num(Math.abs(Number(value)))}` : '—';
const when = value => { if (!value) return 'Unavailable'; const date = new Date(value); return Number.isNaN(date.getTime()) ? String(value) : `${date.toLocaleString(undefined,{year:'numeric',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit',timeZoneName:'short'})}`; };
const age = value => !number(value) ? 'Unknown age' : Number(value) < 0 ? 'Clock mismatch' : Number(value) < 60 ? `${num(value,0)}s` : Number(value) < 3600 ? `${num(Number(value)/60,1)}m` : Number(value) < 86400 ? `${num(Number(value)/3600,1)}h` : `${num(Number(value)/86400,1)}d`;
const safeClass = value => ['fresh','live','stale','error','red','green','amber','neutral','simulated','delayed','closed','invalid','disconnected','critical','warning','info','positive','negative'].includes(String(value)) ? String(value) : 'neutral';
const empty = (title, text='') => `<div class="empty"><strong>${esc(title)}</strong>${text ? `<p>${esc(text)}</p>` : ''}</div>`;
const freshBadge = freshness => {
  const item = typeof freshness === 'string' ? {label:freshness,state:freshness} : freshness || {};
  return `<span class="badge ${safeClass(item.state)}">${esc(item.icon || '◷')} ${esc(item.label || 'Freshness unknown')}</span>`;
};
let snapshot = null;
let sortKey = 'rank';
let sortDirection = 1;
let selectedTicker = null;
let activeRequest = false;
let pollTimer;
const pageInfo = {
  overview:['Overview','The research desk.','Your market, signals and risk. One honest view.'],
  recommendations:['Recommendations','A shortlist, not a promise.','Know the probability, position and plan behind every idea.'],
  portfolio:['Paper portfolio','Watch the risk, too.','Track simulated capital alongside the benchmark.'],
  validation:['Validation','Evidence earns conviction.','The tests, uncertainty and limits behind the model.'],
  alerts:['Alerts','Keep the important things close.','Data issues, changing ranks and risk events.']
};
function route() {
  const page = Object.hasOwn(pageInfo,location.hash.slice(1)) ? location.hash.slice(1) : 'overview';
  const [breadcrumb,title,subtitle] = pageInfo[page];
  $('page-breadcrumb').textContent=breadcrumb; $('page-title').textContent=title; $('page-subtitle').textContent=subtitle;
  document.querySelectorAll('.view').forEach(view => {view.hidden=view.id!==`view-${page}`;});
  document.querySelectorAll('.nav-link').forEach(link => {const active=link.hash===`#${page}`;link.classList.toggle('active',active);if(active)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');});
  document.title=`${breadcrumb} · Stocksel`;
}
function setTheme(theme) {
  document.documentElement.dataset.theme=theme;
  const light=theme==='light';
  $('theme-toggle').innerHTML=`${light?'◐':'☀'} <span>${light?'Dark':'Light'} theme</span>`;
  $('theme-toggle').setAttribute('aria-label',`Switch to ${light?'dark':'light'} theme`);
  try {localStorage.setItem('stocksel-theme',theme);} catch (_) { /* Storage can be unavailable in private contexts. */ }
}
function metric(label,value,note='',tone='',small=false) {
  return `<article class="metric"><div class="metric-label">${esc(label)}</div><div class="metric-value ${safeClass(tone)} ${small?'small-value':''}">${esc(value)}</div><div class="metric-note">${esc(note)}</div></article>`;
}
function dataRow(label,value) {return `<div class="data-row"><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`;}
function quoteFor(ticker) {return (snapshot?.quotes || []).find(quote => quote.ticker===ticker);}
function confidence(rec) {return !snapshot?.validation?.system_validated || snapshot?.mode==='synthetic' || snapshot?.validation?.data_source==='synthetic' ? 'Low · unvalidated' : String(rec.confidence || 'Low');}
function recPrice(rec) {const quote=quoteFor(rec.ticker);return quote?.price ?? rec.price;}
function renderOverview(data) {
  const scan=data.scan || {},feed=data.feed || {},market=data.market || {};
  const mode=data.mode || data.validation?.data_source || 'Unknown';
  const marketLabel=typeof market==='string'?market:market.status || 'Unknown';
  const fresh=typeof data.freshness==='string'?{label:data.freshness,state:data.freshness}:data.freshness || {};
  $('overview-metrics').innerHTML=metric('Eligible universe',num(scan.universe_count ?? scan.n_candidates,0),'Liquidity & size filters applied')+
    metric('Model candidates',num(scan.n_candidates,0),`${(data.recommendations || []).filter(rec => Number(rec.position_pct) > 0).length} sized paper targets`)+
    metric('Market session',marketLabel,'US equities · exchange calendar','',true)+
    metric('Feed freshness',fresh.label || 'Unknown',`${feed.provider || mode} · ${feed.status || 'status unavailable'}`,'',true);
  $('global-freshness').className=`badge ${safeClass(fresh.state)}`;
  $('global-freshness').textContent=`${fresh.icon || '◷'} ${fresh.label || 'Unknown'}`;
  const recs=[...(data.recommendations || [])].sort((a,b)=>(a.rank ?? Infinity)-(b.rank ?? Infinity)).slice(0,5);
  $('top-ideas').innerHTML=recs.length ? '<div class="ideas-row header"><span>TICKER</span><span>PROBABILITY</span><span>POSITION</span><span>CONFIDENCE</span></div>'+recs.map(rec=>`<div class="ideas-row"><div class="ticker-name"><span class="ticker-icon" aria-hidden="true">${esc(String(rec.ticker).slice(0,2))}</span><button type="button" class="ticker-button" data-ticker="${esc(rec.ticker)}">${esc(rec.ticker)}</button></div><div class="ideas-number">${esc(percent(rec.calibrated_prob))}<span class="ideas-secondary">calibrated</span></div><div class="ideas-number">${esc(num(rec.position_pct,2))}%<span class="ideas-secondary">paper target</span></div><div><span class="badge neutral">${esc(confidence(rec).split(' ')[0])}</span></div></div>`).join('') : empty('No recommendations yet','The model scan may still be running, or no names meet the portfolio filters.');
  const simulated=mode==='synthetic' || (data.quotes || []).some(q=>q.simulated);
  $('data-quality').innerHTML='<dl>'+dataRow('Data source',feed.provider || mode)+dataRow('Mode',simulated?'Synthetic historical replay':mode)+dataRow('Quote interval',feed.poll_interval_seconds ? `${feed.poll_interval_seconds}s`:'Provider dependent')+dataRow('Signal date',scan.asof || 'Unavailable')+dataRow('Last model scan',when(scan.scored_at))+dataRow('Next scheduled scan',when(scan.next_scan_at))+dataRow('Market status',marketLabel)+'</dl>';
  $('coverage-note').textContent=feed.coverage || 'Market coverage and entitlements depend on your provider. A connected feed does not establish complete market coverage.';
  const updates=scan.feature_updates || {};
  const descriptions={prices:['Prices & risk alerts','Updated on each quote cycle. Stops are monitoring alerts, not executed orders.'],monthly:['Monthly model signals','Recomputed on their natural schedule. Repeated predictions do not establish additional accuracy.'],fundamentals:['Fundamentals & news','Only information available at the signal date enters the research model.']};
  const entries=Object.entries(updates);
  $('feature-updates').innerHTML=(entries.length?entries:Object.keys(descriptions).map(key=>[key,null])).map(([name,update])=>{
    const updateObj=typeof update==='object' && update!==null?update:{};
    const defaultDescription=descriptions[name] || [name.replaceAll('_',' '),'Scheduled independently of quote refreshes.'];
    const timestamp=typeof update==='string'?update:updateObj.updated_at || updateObj.asof || updateObj.timestamp;
    return `<div class="feature-item"><h3>${esc(updateObj.label || defaultDescription[0])}</h3><p>${esc(updateObj.description || updateObj.cadence || defaultDescription[1])}</p><time>${esc(timestamp?`Updated ${when(timestamp)}`:'Update time unavailable')}</time></div>`;
  }).join('');
}
function renderRecommendations() {
  const all=snapshot?.recommendations || [];
  const query=$('ticker-search').value.trim().toUpperCase(), filter=$('confidence-filter').value;
  const recs=all.filter(rec=>String(rec.ticker).toUpperCase().includes(query) && (filter==='all' || confidence(rec).toLowerCase().startsWith(filter)));
  recs.sort((a,b)=>{
    const av=a[sortKey], bv=b[sortKey];
    if(av===null || av===undefined)return 1;if(bv===null || bv===undefined)return -1;
    return (number(av)&&number(bv)?Number(av)-Number(bv):String(av).localeCompare(String(bv)))*sortDirection;
  });
  $('recommendation-count').textContent=`${recs.length} of ${all.length} ideas`;
  document.querySelectorAll('.sort').forEach(button=>{const selected=button.dataset.sort===sortKey;button.setAttribute('aria-pressed',String(selected));button.closest('th').setAttribute('aria-sort',selected?(sortDirection===1?'ascending':'descending'):'none');});
  $('recommendations-body').innerHTML=recs.length?recs.map(rec=>{
    const quote=quoteFor(rec.ticker),price=recPrice(rec);
    const priceLabel=number(price)?amount(price):'No quote';
    const rankChange=number(rec.previous_rank)&&number(rec.rank)?Number(rec.previous_rank)-Number(rec.rank):null;
    const movement=rankChange===null?'No previous rank':rankChange===0?'Unchanged':`${rankChange>0?'↑':'↓'} ${Math.abs(rankChange)}`;
    return `<tr><td><button type="button" class="ticker-button" data-ticker="${esc(rec.ticker)}">${esc(rec.ticker)}</button><span class="secondary">Signals ${esc(rec.signal_asof || snapshot.scan?.asof || 'unavailable')}</span></td><td>${esc(num(rec.rank,0))}<span class="secondary">${esc(movement)}</span></td><td>${esc(percent(rec.calibrated_prob))}<span class="secondary">Haircut: ${esc(percent(rec.prob_after_haircut))}</span></td><td>${esc(signedPercent(rec.expected_excess_after_haircut))}</td><td>${esc(number(rec.position_pct)?`${num(rec.position_pct)}%`:'—')}</td><td>${esc(priceLabel)}<span class="secondary">${quote?`${freshBadge(quote.freshness)} · ${esc(age(quote.age_seconds))}`:'Quote unavailable'}</span></td><td>${esc(rec.entry_date || '—')}<span class="secondary">Exit ${esc(rec.planned_exit || '—')}</span></td><td>${esc(amount(rec.stop))}</td><td class="signals-cell">${esc(rec.top3_signals_shap || 'Unavailable')}</td><td><span class="badge neutral">${esc(confidence(rec))}</span></td></tr>`;
  }).join(''):`<tr><td colspan="10">${empty(all.length?'No matching ideas':'No recommendations available',all.length?'Try a different ticker or confidence filter.':'A completed model scan and eligible candidates are required.')}</td></tr>`;
}
function renderPortfolio(data) {
  const portfolio=data.portfolio || {};
  $('portfolio-metrics').innerHTML=metric('Paper equity',amount(portfolio.equity),'Marked using available provider quotes')+metric('Total paper P&L',amount(portfolio.pnl),`${signedPercent(portfolio.pnl_pct)} since inception`,Number(portfolio.pnl)>=0?'positive':'negative')+metric('Current drawdown',percent(portfolio.drawdown),'From the recorded equity high-water mark',Number(portfolio.drawdown)<0?'negative':'')+metric('Benchmark return',signedPercent(portfolio.benchmark_return),'Same recorded paper period');
  const positions=Array.isArray(portfolio.positions)?portfolio.positions:Object.entries(portfolio.positions || {}).map(([ticker,value])=>({ticker,...value}));
  $('positions-body').innerHTML=positions.length?positions.map(position=>`<tr><td><strong>${esc(position.ticker)}</strong></td><td>${esc(num(position.quantity ?? position.shares,4))}</td><td>${esc(amount(position.entry_price ?? position.entry_px ?? position.avg_cost))}</td><td>${esc(amount(position.price ?? quoteFor(position.ticker)?.price))}</td><td>${esc(amount(position.market_value ?? position.value))}</td><td>${esc(amount(position.pnl ?? position.unrealized_pnl))}</td></tr>`).join(''):`<tr><td colspan="6">${empty('No paper positions recorded','A model recommendation is not an executed paper position.')}</td></tr>`;
  renderChart(portfolio.history || []);
}
function renderChart(history) {
  const rows=history.filter(row=>number(row.equity)||number(row.benchmark));
  $('equity-table').innerHTML=rows.length?`<table><thead><tr><th scope="col">Date</th><th scope="col">Paper equity</th><th scope="col">Benchmark</th></tr></thead><tbody>${rows.map(row=>`<tr><td>${esc(row.date)}</td><td>${esc(amount(row.equity))}</td><td>${esc(amount(row.benchmark))}</td></tr>`).join('')}</tbody></table>`:'No equity history is available.';
  if(rows.length<2){$('equity-chart').innerHTML=empty('Waiting for an equity history','At least two paper valuation dates are needed to draw a performance chart.');return;}
  const width=1000,height=290,pad={left:78,right:25,top:22,bottom:43};
  const values=rows.flatMap(row=>[row.equity,row.benchmark]).filter(number).map(Number);
  const min=Math.min(...values),max=Math.max(...values),spread=(max-min)||Math.abs(max)*.01||1;
  const low=min-spread*.12,high=max+spread*.12;
  const x=index=>pad.left+index/(rows.length-1)*(width-pad.left-pad.right);
  const y=value=>height-pad.bottom-(Number(value)-low)/(high-low)*(height-pad.top-pad.bottom);
  const path=key=>{let drawing=false;return rows.map((row,index)=>{if(!number(row[key])){drawing=false;return '';}const point=`${drawing?'L':'M'}${x(index).toFixed(2)},${y(row[key]).toFixed(2)}`;drawing=true;return point;}).join(' ');};
  const ticks=[0,1,2,3,4].map(i=>{const value=low+(high-low)*i/4;return `<line class="grid" x1="${pad.left}" x2="${width-pad.right}" y1="${y(value)}" y2="${y(value)}"/><text x="${pad.left-12}" y="${y(value)+4}" text-anchor="end">${esc(Math.abs(value)>=1000?`${num(value/1000,1)}k`:num(value,0))}</text>`;}).join('');
  const dateTicks=[...new Set([0,Math.floor((rows.length-1)/2),rows.length-1])].map(index=>`<text x="${x(index)}" y="${height-10}" text-anchor="${index===0?'start':index===rows.length-1?'end':'middle'}">${esc(String(rows[index].date).slice(0,10))}</text>`).join('');
  // Individual points expose native tooltips and keyboard-focusable accessible labels.
  const pointStep=Math.max(1,Math.ceil(rows.length/80));
  const points=rows.map((row,index)=>number(row.equity)&&(index%pointStep===0||index===rows.length-1)?`<circle class="chart-point" cx="${x(index)}" cy="${y(row.equity)}" r="3" tabindex="0" role="img" aria-label="${esc(`${row.date}: paper equity ${amount(row.equity)}, benchmark ${amount(row.benchmark)}`)}"><title>${esc(`${row.date}\nPaper: ${amount(row.equity)}\nBenchmark: ${amount(row.benchmark)}`)}</title></circle>`:'').join('');
  $('equity-chart').innerHTML=`<svg class="equity-svg" viewBox="0 0 ${width} ${height}" role="group" aria-labelledby="equity-chart-title equity-chart-description"><title id="equity-chart-title">Paper equity versus benchmark</title><desc id="equity-chart-description">${esc(`${rows[0].date} to ${rows.at(-1).date}. Teal solid line: paper portfolio. Blue dashed line: benchmark. Values in account currency. Focus points for exact values, or expand the data table below.`)}</desc>${ticks}<path class="benchmark-line" d="${path('benchmark')}"/><path class="strategy-line" d="${path('equity')}"/>${points}${dateTicks}</svg>`;
}
function renderValidation(data) {
  const validation=data.validation || {},metrics=validation.metrics || {};
  const holdout=String(validation.holdout_status || 'Unavailable').replaceAll('_',' ');
  $('validation-source').textContent=`${validation.data_source || data.mode || 'Unknown source'} data`;
  $('validation-metrics').innerHTML=metric('Deflated Sharpe probability',percent(validation.dsr),'Gate: ≥ 95% after multiple-testing correction','',true)+metric('Backtest overfit probability',percent(validation.pbo),'Gate: ≤ 50% · all attempted trials count','',true)+metric('Holdout status',holdout,'Previously opened data cannot be reused as fresh','',true)+metric('Registered trials',num(validation.n_trials,0),'Failed configurations count, too');
  const flags=validation.red_flags || [];
  $('validation-flags').innerHTML=flags.length?flags.map(flag=>{const item=typeof flag==='string'?{message:flag,level:'warning'}:flag;return `<div class="flag ${safeClass(item.level)}"><span class="flag-icon" aria-hidden="true">${item.level==='critical'?'!':'◇'}</span><p>${esc(item.message || item.reason || 'Unspecified validation flag')}</p></div>`;}).join(''):empty('No validation flags reported','Absence of reported flags does not establish a validated strategy.');
  const fields=[['Net CAGR','cagr',percent],['Annual excess return','excess_ann',signedPercent],['Net Sharpe','sharpe',v=>num(v)],['Maximum drawdown','max_drawdown',percent],['Calmar ratio','calmar',v=>num(v)],['Sortino ratio','sortino',v=>num(v)],['Information ratio','information_ratio',v=>num(v)],['Annual turnover','turnover_ann',v=>number(v)?`${num(v)}×`:'—'],['Annual cost drag','cost_drag_ann',percent]];
  $('research-metrics').innerHTML=fields.map(([label,key,formatter])=>dataRow(label,formatter(metrics[key]))).join('');
  let decay=validation.signal_decay || [];
  if(!Array.isArray(decay))decay=Object.entries(decay).map(([signal,value])=>({signal,...(typeof value==='object'?value:{status:value})}));
  $('signal-decay').innerHTML=decay.length?`<div class="decay-grid">${decay.map(row=>`<div class="decay-item"><div><strong>${esc(row.signal || row.feature || row.name || 'Unnamed signal')}</strong><span class="muted">IC ${esc(num(row.ic ?? row.mean_ic,3))} · t ${esc(num(row.tstat ?? row.t_stat))}</span></div><span class="badge ${row.status==='alive'?'fresh':'neutral'}">${esc(row.status || 'Not evaluated')}</span></div>`).join('')}</div>`:empty('Signal health not yet evaluated','A completed research run provides rolling IC and decay diagnostics.');
}
function renderAlerts(data) {
  const alerts=data.alerts || [];
  $('alert-count').textContent=String(alerts.length);
  $('alerts-list').innerHTML=alerts.length?[...alerts].reverse().map(alert=>`<article class="alert ${safeClass(alert.level)}"><span class="alert-icon" aria-hidden="true">${alert.level==='critical' || alert.level==='error'?'!':'◇'}</span><div class="alert-content"><span class="alert-level">${esc(alert.level || 'info')}${alert.ticker?` · ${esc(alert.ticker)}`:''}</span><p>${esc(alert.message)}</p><time>${esc(when(alert.time))}</time></div></article>`).join(''):empty('No alerts in this snapshot','Monitoring continues while the application is running.');
}
function render(data) {
  snapshot=data;
  $('updated-at').textContent=when(data.updated_at);
  if(data.updated_at)$('updated-at').dateTime=data.updated_at;
  $('loading').hidden=true;$('content').hidden=false;
  const validated=data.validation?.system_validated && data.mode!=='synthetic' && data.validation?.data_source!=='synthetic';
  $('validation-notice').hidden=Boolean(validated);
  if(data.disclaimer)$('disclaimer').textContent=data.disclaimer;
  renderOverview(data);renderRecommendations();renderPortfolio(data);renderValidation(data);renderAlerts(data);route();
  if(selectedTicker && $('stock-detail').open)renderDetail(selectedTicker);
}
function renderDetail(ticker) {
  const rec=(snapshot?.recommendations || []).find(row=>row.ticker===ticker);
  if(!rec)return;
  const quote=quoteFor(ticker),scan=snapshot.scan || {};
  $('detail-title').textContent=ticker;
  const quoteDetails=quote?dataRow('Current quote',amount(quote.price))+dataRow('Data timestamp',when(quote.data_timestamp))+dataRow('Received at',when(quote.received_at))+dataRow('Data age',age(quote.age_seconds))+dataRow('Receive age',age(quote.receive_age_seconds))+dataRow('Provider delay',number(quote.documented_delay_seconds)?age(quote.documented_delay_seconds):'Unknown')+dataRow('Provider',quote.provider || snapshot.feed?.provider || 'Unknown')+dataRow('Quote market session',typeof quote.market_status==='object'?(quote.market_status?.status || quote.market_status?.state):quote.market_status || snapshot.market?.status || 'Unknown')+dataRow('Simulated',quote.simulated?'Yes · historical replay':'No'):'<p>No price quote is available for this ticker.</p>';
  $('detail-content').innerHTML=`<div class="detail-grid"><div class="detail-stat"><span>Calibrated probability</span><strong>${esc(percent(rec.calibrated_prob))}</strong></div><div class="detail-stat"><span>Probability after haircut</span><strong>${esc(percent(rec.prob_after_haircut))}</strong></div><div class="detail-stat"><span>Expected excess after haircut</span><strong>${esc(signedPercent(rec.expected_excess_after_haircut))}</strong></div><div class="detail-stat"><span>Target position</span><strong>${esc(number(rec.position_pct)?`${num(rec.position_pct)}%`:'—')}</strong></div></div><div class="detail-section"><h3>Position plan</h3><dl>${dataRow('Entry',rec.entry_date || 'Unavailable')}${dataRow('Planned exit',rec.planned_exit || 'Unavailable')}${dataRow('Stop level',amount(rec.stop))}${dataRow('Signal reference close',amount(rec.ref_close))}${dataRow('Confidence',confidence(rec))}</dl></div><div class="detail-section"><h3>Quote freshness ${quote?freshBadge(quote.freshness):''}</h3><dl>${quoteDetails}</dl></div><div class="detail-section"><h3>Signal provenance</h3><dl>${dataRow('Signal date',rec.signal_asof || scan.asof || 'Unavailable')}${dataRow('Signal age',age(rec.signal_age_seconds))}${dataRow('Model scan received',when(scan.scored_at))}${dataRow('Rank',num(rec.rank,0))}${dataRow('Previous rank',num(rec.previous_rank,0))}</dl><p>Signal timestamps and quote timestamps describe different clocks. A fresh quote does not make an older monthly signal real-time.</p></div><div class="detail-section"><h3>Top 3 attributed signals</h3><p>${esc(rec.top3_signals_shap || 'No feature attributions are available.')}</p></div><div class="detail-notice">${esc(confidence(rec))}. Model targets are research outputs, not financial advice or executed orders. Synthetic results do not establish real-market profitability.</div>`;
}
function markSnapshotDisconnected() {
  if(!snapshot)return;
  const cached=structuredClone(snapshot);
  cached.freshness={state:'red',label:'Cached snapshot · connection lost',icon:'!'};
  cached.feed={...cached.feed,status:'disconnected'};
  cached.quotes=(cached.quotes || []).map(quote=>({...quote,
    age_seconds:Number.isNaN(new Date(quote.data_timestamp).getTime())?null:Math.max(0,(Date.now()-new Date(quote.data_timestamp).getTime())/1000),
    receive_age_seconds:Number.isNaN(new Date(quote.received_at).getTime())?null:Math.max(0,(Date.now()-new Date(quote.received_at).getTime())/1000),
    freshness:{state:'red',label:'Cached quote · connection lost',icon:'!',live:false}}));
  render(cached);
}
async function refresh() {
  if(activeRequest)return;
  clearTimeout(pollTimer);activeRequest=true;$('refresh').disabled=true;
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),12000);
  try {
    const response=await fetch('/api/snapshot',{signal:controller.signal,headers:{'Accept':'application/json'},cache:'no-store'});
    if(!response.ok)throw new Error(`Snapshot request failed (${response.status})`);
    const data=await response.json();
    if(!data || typeof data!=='object' || Array.isArray(data))throw new Error('Unexpected snapshot response');
    if(data.status==='error')throw new Error(data.error || 'The research worker reported an error');
    $('network-error').hidden=true;$('connection').className='connection connected';$('connection').innerHTML='<span class="dot" aria-hidden="true"></span> Workspace connected';
    if(data.status==='loading'){
      if(!snapshot){$('loading').hidden=false;$('loading').querySelector('p').textContent=data.message || 'Preparing data and computing the model snapshot. The interface will update automatically.';}
    } else render(data);
  } catch(error) {
    markSnapshotDisconnected();
    $('connection').className='connection failed';$('connection').innerHTML='<span class="dot" aria-hidden="true"></span> Snapshot unavailable';
    $('network-error').textContent=`${error.name==='AbortError'?'The snapshot request timed out.':error.message} ${snapshot?'Showing the last received snapshot; its freshness may have changed.':'Waiting for a successful snapshot.'} Retrying automatically.`;
    $('network-error').hidden=false;
    if(!snapshot)$('loading').hidden=true;
  } finally {clearTimeout(timeout);activeRequest=false;$('refresh').disabled=false;pollTimer=setTimeout(refresh,5000);}
}
$('ticker-search').addEventListener('input',renderRecommendations);
$('confidence-filter').addEventListener('change',renderRecommendations);
$('refresh').addEventListener('click',refresh);
$('theme-toggle').addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
$('detail-close').addEventListener('click',()=>$('stock-detail').close());
$('stock-detail').addEventListener('click',event=>{if(event.target===$('stock-detail')){const bounds=event.target.getBoundingClientRect();if(event.clientX<bounds.left || event.clientX>bounds.right || event.clientY<bounds.top || event.clientY>bounds.bottom)event.target.close();}});
$('stock-detail').addEventListener('close',()=>{selectedTicker=null;});
document.addEventListener('click',event=>{
  const tickerButton=event.target.closest('[data-ticker]');
  if(tickerButton){selectedTicker=tickerButton.dataset.ticker;renderDetail(selectedTicker);$('stock-detail').showModal();return;}
  const sortButton=event.target.closest('[data-sort]');
  if(sortButton){if(sortKey===sortButton.dataset.sort)sortDirection*=-1;else{sortKey=sortButton.dataset.sort;sortDirection=sortKey==='ticker'||sortKey==='rank'?1:-1;}renderRecommendations();}
});
window.addEventListener('hashchange',route);
// Keep the theme reachable on mobile without changing navigation or order.
const themeButton=$('theme-toggle');
const mobileTheme=()=>{if(window.matchMedia('(max-width:900px)').matches){document.querySelector('.sidebar').append(themeButton);themeButton.classList.add('mobile-theme');themeButton.style.width='auto';themeButton.style.marginTop='0';}else{document.querySelector('.sidebar-bottom').append(themeButton);themeButton.classList.remove('mobile-theme');themeButton.style.width='';themeButton.style.marginTop='';}};
window.addEventListener('resize',mobileTheme);
try{setTheme(localStorage.getItem('stocksel-theme') || (matchMedia('(prefers-color-scheme:light)').matches?'light':'dark'));}catch(_){setTheme('dark');}
route();mobileTheme();refresh();
