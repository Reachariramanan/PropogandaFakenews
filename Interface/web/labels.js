/* Display names and glossary for the Evidence Desk.
   One place maps stored codes (sites, languages, channels, countries, verdicts) to the words a reader should see,
   and holds the definitions behind every (?) tooltip. Loaded before all other scripts; no dependencies. */
(function () {
  'use strict';
  var esc = function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); };

  /* ---------- sites, sources and platforms ---------- */
  var SITE = {
    quint_webqoof: 'The Quint WebQoof', pib_factcheck: 'PIB Fact Check', vishvasnews: 'Vishvas News', boomlive: 'BOOM',
    snopes: 'Snopes', fullfact: 'Full Fact', factly: 'Factly', altnews: 'Alt News', dfrac: 'DFRAC', newsmeter: 'NewsMeter',
    factcheck_org: 'FactCheck.org', 'factcheck.org': 'FactCheck.org', 'politifact.com': 'PolitiFact', politifact: 'PolitiFact',
    newschecker: 'Newschecker', euvsdisinfo: 'EUvsDisinfo', bellingcat: 'Bellingcat', indiatoday: 'India Today',
    reddit_posts: 'Reddit', twitter_posts: 'X (Twitter)', telegram_posts: 'Telegram', mastodon_posts: 'Mastodon',
    facebook_posts: 'Facebook', linkedin_posts: 'LinkedIn', instagram_posts: 'Instagram', whatsapp_posts: 'WhatsApp',
    reddit: 'Reddit', twitter: 'X (Twitter)', x: 'X (Twitter)', telegram: 'Telegram', mastodon: 'Mastodon', facebook: 'Facebook',
    linkedin: 'LinkedIn', instagram: 'Instagram', whatsapp: 'WhatsApp', youtube: 'YouTube', web: 'Web', tiktok: 'TikTok'
  };
  function titleCase(s) { return String(s).replace(/[_\-.]+/g, ' ').trim().replace(/\b[a-z]/g, function (c) { return c.toUpperCase(); }); }
  function site(code) {
    if (code == null || String(code).trim() === '') return 'Unknown';
    var k = String(code).trim().toLowerCase();
    return SITE[k] || titleCase(k);
  }
  /* social platforms stored as <name>_posts are not fact-checkers */
  function isSocial(code) { return /_posts$/i.test(String(code || '')); }
  function channel(code) { return code ? site(code) : 'Web'; }
  /* Evidence Base platform labels: the realtime cache stores a short platform code ("twitter") and the sync map a table name ("twitter_posts"); both read the same */
  var PLATFORM = {
    news: 'News', scraped_articles: 'News', scraped_articles_trusted: 'News (trusted sources)',
    reddit: 'Reddit', reddit_posts: 'Reddit', twitter: 'X (Twitter)', twitter_posts: 'X (Twitter)', x: 'X (Twitter)', x_posts: 'X (Twitter)',
    telegram: 'Telegram', telegram_posts: 'Telegram', mastodon: 'Mastodon', mastodon_posts: 'Mastodon', instagram: 'Instagram', instagram_posts: 'Instagram',
    tiktok: 'TikTok', tiktok_posts: 'TikTok', snapchat: 'Snapchat', snapchat_posts: 'Snapchat', facebook: 'Facebook', facebook_posts: 'Facebook', facebook_profile_posts: 'Facebook',
    linkedin: 'LinkedIn', linkedin_posts: 'LinkedIn', baidu_tieba: 'Baidu Tieba', baidu_tieba_posts: 'Baidu Tieba', discord: 'Discord', discord_posts: 'Discord',
    whatsapp: 'WhatsApp', whatsapp_posts: 'WhatsApp', pinterest: 'Pinterest', pinterest_posts: 'Pinterest', truthsocial: 'Truth Social', truthsocial_posts: 'Truth Social', youtube: 'YouTube', youtube_comments: 'YouTube'
  };
  /* the stored table behind a platform code: 'twitter' and 'x' are two different source tables that both read 'X (Twitter)' */
  function platformTable(code) {
    if (code == null || String(code).trim() === '') return '';
    var k = String(code).trim().toLowerCase();
    return k === 'news' ? 'scraped_articles' : /_posts$|^scraped_|_comments$/.test(k) ? k : k + '_posts';
  }
  function platform(code) {
    if (code == null || String(code).trim() === '') return 'Unknown platform';
    var k = String(code).trim().toLowerCase();
    return PLATFORM[k] || titleCase(k.replace(/_posts$/, ''));
  }

  /* ---------- languages ---------- */
  var LANG = {
    hi: 'Hindi', en: 'English', te: 'Telugu', kn: 'Kannada', gu: 'Gujarati', ta: 'Tamil', bn: 'Bengali', mr: 'Marathi', pa: 'Punjabi',
    as: 'Assamese', or: 'Odia', ml: 'Malayalam', ur: 'Urdu', ne: 'Nepali', pt: 'Portuguese', si: 'Sinhala', fr: 'French', tl: 'Tagalog',
    es: 'Spanish', it: 'Italian', de: 'German', ru: 'Russian', ar: 'Arabic', zh: 'Chinese', ja: 'Japanese', ko: 'Korean', id: 'Indonesian',
    tr: 'Turkish', uk: 'Ukrainian', fa: 'Persian', he: 'Hebrew', th: 'Thai', vi: 'Vietnamese', nl: 'Dutch', pl: 'Polish', sv: 'Swedish',
    sd: 'Sindhi', sa: 'Sanskrit', my: 'Burmese', ps: 'Pashto'
  };
  var LANGNAME = {}; Object.keys(LANG).forEach(function (k) { LANGNAME[LANG[k].toLowerCase()] = LANG[k]; });
  LANGNAME.oriya = 'Odia'; LANGNAME.panjabi = 'Punjabi';
  function lang(code) {
    if (code == null) return 'Unknown';
    var k = String(code).trim().toLowerCase();
    if (!k || k === 'und' || k === 'unknown' || k === 'null' || k === 'undefined' || k === 'n/a' || k === 'mul' || k === 'zxx') return 'Unknown';
    var base = k.split(/[-_]/)[0];
    return LANG[base] || LANGNAME[k] || LANGNAME[base] || titleCase(k);
  }

  /* ---------- countries (ISO 3166 alpha-2 used by the claim index; the Lab sends names itself) ---------- */
  var COUNTRY = { IN: 'India', US: 'United States', GB: 'United Kingdom', EU: 'European Union', NL: 'Netherlands', PK: 'Pakistan', CN: 'China',
    RU: 'Russia', UA: 'Ukraine', BD: 'Bangladesh', LK: 'Sri Lanka', NP: 'Nepal', AU: 'Australia', CA: 'Canada', DE: 'Germany', FR: 'France' };
  function country(code) { if (!code) return 'Unknown'; var k = String(code).toUpperCase(); return COUNTRY[k] || k; }

  /* ---------- verdicts and source types ---------- */
  var VERDICT = { FAKE: 'Fake', MISLEADING: 'Misleading', LEGIT: 'Legit', INCONCLUSIVE: 'Inconclusive', UNRATED: 'Unrated', PROPAGANDA: 'Propaganda' };
  function verdict(v) { return VERDICT[v] || (v ? titleCase(v) : 'Unrated'); }
  var TIER = { ifcn: 'IFCN signatory', independent: 'Independent', government: 'Government unit', osint: 'OSINT investigators' };
  function tier(t) { return TIER[t] || titleCase(t || 'Unknown'); }

  /* how a score or rating was arrived at, as stored by the index builder */
  var SCORE_SOURCE = { blend: 'Blend of fact-checker label and verdict engine', engine: 'Verdict engine only', 'fact-checker-label': 'Fact-checker label only', none: 'No score' };
  function scoreSource(c) { return SCORE_SOURCE[c] || (c ? titleCase(c) : 'No score'); }
  function verdictBasis(b) {
    if (!b || b === 'claimreview') return 'Stated by the fact-checker';
    if (/^llm/i.test(b)) return 'Inferred by a language model from the article';
    return 'Inferred from ' + String(b).replace(/[_\-]+/g, ' ');
  }

  /* ---------- dates ---------- */
  var MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  /* "27 Oct 2026" from any ISO string or Date, in UTC so the label matches the stored date */
  function dateLong(s, noYear) {
    var d = s instanceof Date ? s : new Date(s);
    if (!s || isNaN(d)) return 'unknown date';
    return d.getUTCDate() + ' ' + MON[d.getUTCMonth()] + (noYear ? '' : ' ' + d.getUTCFullYear());
  }
  function dateTimeUTC(s) {
    var d = new Date(s); if (!s || isNaN(d)) return 'unknown';
    return dateLong(d) + ', ' + d.toISOString().slice(11, 16) + ' UTC';
  }

  /* engine technique codes ("Name_Calling,Labeling") as readable names ("Name Calling / Labeling"); these match the glossary keys */
  function technique(t) { return String(t == null ? 'Unknown' : t).replace(/_/g, ' ').replace(/,/g, ' / '); }
  /* the verdict as a plain sentence for Claim Check */
  var PLAIN = { FAKE: 'Likely false', MISLEADING: 'Likely misleading', LEGIT: 'Likely accurate', INCONCLUSIVE: 'Inconclusive', PROPAGANDA: 'Propaganda-style wording', UNRATED: 'Unrated' };
  function verdictPlain(v) { return PLAIN[v] || verdict(v); }

  /* ---------- glossary ---------- */
  var GLOSSARY = {
    'claim check': { label: 'Claim Check', definition: 'The two-step check for text you paste. Step 1 asks whether the claim looks fake or misleading. Step 2, always shown, rates the 14 propaganda techniques in its wording and highlights them. Propaganda describes how a text is worded, whether or not it is true.' },
    'plausibility score': { label: 'Plausibility score', definition: 'Estimated probability, from 0 to 1, that a claim is fake or misleading. It blends the fact-checker\'s own label with the verdict engine\'s estimate. Bands: below 0.5, 0.5 to 0.8, 0.8 and above. A triage signal, not a ruling.' },
    'engine probability': { label: 'Engine probability', definition: 'The verdict engine\'s own estimate that the claim is fake or misleading, from the nearest fact-checks plus a language-model check. Shown next to the fact-checker\'s label so the two can be compared.' },
    'lift': { label: 'Lift', definition: 'How much more often posts about a country carry propaganda techniques than expected for posts of the same length. 1.0 means as expected, 2.0 means twice as often. Adjusted for post length, so countries are comparable.' },
    'rate': { label: 'Rate', definition: 'Share of a country\'s posts that contain at least one strong propaganda span. The raw rate over-fires on short or emotive posts, so compare countries by lift rather than rate.' },
    'min_n': { label: 'Minimum posts (min_n)', definition: 'The smallest number of posts, currently 30, a country needs before its rate is coloured on the map or ranked as reliable. Below it the figures swing too much to trust.' },
    'model estimate': { label: 'Model estimate', definition: 'Used when no fact-check in the index matches closely. The verdict engine reads the claim and gives its own estimate. It is a model\'s judgement, not a fact-checker\'s ruling, and its accuracy on real data has not been measured.' },
    'second opinion': { label: 'Second opinion', definition: 'When a fact-check matches closely, its rating leads. The verdict engine also reads the claim and its answer is shown beside it, so you can see whether they agree. Disagreement is a reason to read the fact-check itself.' },
    'span tagger': { label: 'Span tagger', definition: 'The step-2 model that finds the exact words in a text that use one of 14 propaganda techniques. It reads the raw text and returns each technique with its position and a confidence. Not validated on social media.' },
    'fine-tuned laya check': { label: 'Fine-tuned Laya check', definition: 'Laya is the fine-tuned model that reads a structured summary of the text (the extracted claim, signals and the span tagger\'s spans), not the raw words. Step 2 shows its own five-class read and one raw propaganda yes/no output. Because its input already contains the tagger\'s spans it is not an independent second opinion, it is not validated on social media, and it never decides step 1.' },
    'propaganda signal': { label: 'Propaganda signal', definition: 'A one-line summary of the span tagger only: STRONG if any span is rated strong (the engine rates a span strong when its mean probability is at least its class threshold and 0.50), WEAK if spans were reported but none is strong, NONE if no span was reported. Not calibrated and not validated on social media. Laya is shown separately and is not merged into it.' },
    'laya propaganda head': { label: 'Laya propaganda head', definition: 'A raw yes/no output of the Laya model (0 to 1). FNP measured it at 1.0 on news-article states and 0.0 on others, so it appears to separate article-style training data rather than propaganda wording (it still read 0.88 on one loaded short sentence). It is a raw output, not a verdict, not validated on social media, and is not used for the summary line or step 1.' },
    'one-sided': { label: 'One-sided', definition: 'Gemma\'s own judgement of how one-sided the framing is, from 0 (balanced) to 1 (one position only, opponents caricatured). Unvalidated: a 0 can also mean nothing was returned, so it never proves balance. An indicator of framing, not proof that the claim is false.' },
    'secondary signal': { label: 'Secondary signal', definition: 'A signal shown for context that does not change the verdict. It can be wrong on its own, so read it alongside the main result.' },
    'span encoder': { label: 'Span encoder', definition: 'The model that marks the exact words in a post that use one of 14 propaganda techniques. It is a signal for triage and has not been validated on social media.' },
    'Gemma check': { label: 'Gemma check', definition: 'A second opinion from the Gemma language model on a sample of posts: whether it agrees the flagged technique is present, or whether the post sits near a known fact-check. Pending until the sample has been run.' },
    'from the database': { label: 'From the database', definition: 'A value the Realtime Database stores for this item (platform, source, language, time, topic, engagement). It is shown as it was synced, with no interpretation by FNP.' },
    'inferred by fnp': { label: 'Inferred by FNP', definition: 'An estimate made by the FNP analysis pipeline from the item\'s text, such as the propaganda signal, techniques, countries or a verdict. It is not stored in the database, can be wrong, and exists only for items FNP has analysed.' },
    'batch cheap pass': { label: 'Batch cheap pass', definition: 'The fast analysis FNP runs in bulk and exports run by run: a propaganda signal, a probability for each of 14 techniques, countries and the nearest known fact-check. It is cheap, but it has not reached every item yet (the export fills in over several runs) and it is shallower than a deep verdict. Items too short to score get no signal.' },
    'tier': { label: 'Tier', definition: 'A ranking aid in the Triage tab: High, Medium or Low says how strong FNP\'s signals are for an item, so you know what to read first. It is not a verdict and has not been validated against posts that people have labelled.' },
    'triage queue': { label: 'Triage queue', definition: 'The ranked list of items in the Triage tab. The FNP export lists posts FNP has already looked at closely; the Realtime Database queue ranks stored posts and articles by FNP\'s batch signals and deep verdicts. Items FNP has not scored yet are not in it.' },
    'validated': { label: 'Validated', definition: 'Checked by a person against the original and the fact-check. Today no triage item is validated, so every tier is a first guess for ordering your reading, not a result.' },
    'deep verdict': { label: 'Deep verdict', definition: 'The result of a full Claim Check run on one item: the engine\'s fake-or-not call with a probability for each class, plus the 14 propaganda technique bars. It takes 15 to 40 seconds per item, so it exists only for the items that were analysed in a scheduled run or by hand in the Workbench.' },
    'partial result': { label: 'Partial result', definition: 'A deep run whose verdict timed out in the engine. The propaganda signals (technique bars and highlighted wording) are kept, but there is no verdict, so it is never counted in a verdict share. The scheduled analysis retries such an item on later runs, at most 3 times.' },
    'scheduled analysis': { label: 'Scheduled analysis', definition: 'A bounded deep analysis the server runs by itself every few hours, a few dozen items at a time, one after another, newest and most suspicious first. It stops early when the shared engine is busy and cannot be started from this page. When it is off, deep verdicts come only from manual Workbench runs.' },
    'analysed coverage': { label: 'Analysed coverage', definition: 'The share of items in view that FNP has analysed so far. Every figure marked Inferred by FNP describes only those items, so read it next to this share.' },
    'database classification': { label: 'Database classification', definition: 'The topic, category and subtopic the Realtime Database assigned to an item. It is the database\'s own labelling, not FNP\'s. Only news is classified; everything else is Unclassified.' },
    'unclassified': { label: 'Unclassified', definition: 'The database has not assigned a topic to this item. All social posts are unclassified, as are news articles the classifier has not reached yet. It does not mean the item is off-topic.' },
    'near-duplicate': { label: 'Near-duplicate', definition: 'Another item whose text is almost identical, found by comparing text fingerprints (within 3 differing bits). Typical for reposts and wire stories. The database\'s own duplicate flag is separate and may differ.' },
    'uncalibrated signal': { label: 'Uncalibrated signal', definition: 'A score that ranks items but has not been checked against real outcomes. A p_prop of 0.8 means the item ranks high for propaganda-style wording, not that 80% of such items are propaganda. Use it to decide what to read first.' },
    'workbench': { label: 'Workbench', definition: 'The Evidence Base tab where you pick up to 25 stored items, run Claim Check on them one after another and read each result. Nothing is analysed unless you press Run.' },
    'selection': { label: 'Selection', definition: 'The items you have picked in the Workbench, at most 25. Claim Check, the comparison with the Fake News Database and Insights for selection all work on exactly this set. It lives in this browser tab; save it under a name to bring it back later.' },
    'grounded insight': { label: 'Grounded insight', definition: 'A statement that cites only rows or numbers from the connected database, with the count and the group it is divided by. No outside knowledge is used, and a Gemma paraphrase is only shown after the server has checked every number and every cited row against the same facts.' },
    'degraded': { label: 'Degraded', definition: 'The verdict engine ran out of time (about 55 s) on the deep check and returned only its fast result and the propaganda techniques. It is shown with a warning, never as a normal verdict, and it is not stored, so you can retry it.' },
    'cached result': { label: 'Cached result', definition: 'A Claim Check result stored by an earlier run of the same item text. It is returned instantly without calling the engine. The badge shows when it was stored; use Ignore stored results to run the engine again.' },
    'technique probability': { label: 'Technique probability', definition: 'The bar for each of the 14 propaganda techniques. Until the engine reports its own estimate, it is the highest confidence of any span tagged with that technique, and zero when none was found. It is an uncalibrated ranking aid, not the chance that the text really uses the technique.' },
    'INCONCLUSIVE': { label: 'Inconclusive', definition: 'The fact-checker could not show the claim to be true or false, for example unproven, unverifiable or needs context.' },
    'MISLEADING': { label: 'Misleading', definition: 'Partly true, missing context, or presented in a way that leads to a wrong conclusion.' },
    'FAKE': { label: 'Fake', definition: 'Rated false, fabricated or manipulated by the fact-checker.' },
    'LEGIT': { label: 'Legit', definition: 'Rated true or accurate by the fact-checker.' },
    'PROPAGANDA': { label: 'Propaganda', definition: 'A secondary engine tag: the wording uses persuasion techniques, whether or not the facts are right. Unreliable on short social posts.' },
    'score source': { label: 'Score source', definition: 'Where the plausibility score comes from: the fact-checker\'s label, the verdict engine, or a blend of both.' },
    'similarity': { label: 'Similarity', definition: 'How close a past fact-check is to this text, from 0 to 1, using text vectors. Around 0.7 or more usually means the same claim; Claim Check ignores anything below 0.25.' },
    'label score': { label: 'Fact-checker label score', definition: 'The 0 to 1 value the index gives the fact-checker\'s rating: high for fake, middling for misleading or inconclusive, low for legit. When the rating was inferred rather than stated, it is pulled toward 0.5.' },
    'blend': { label: 'Blend method', definition: 'When both the fact-checker\'s label and the verdict engine have a score, the plausibility score is their simple average (50% each). The weights are fixed, not fitted.' },
    'calibration': { label: 'Calibration', definition: 'Whether a score of 0.8 really means about 80% of such claims are false. These scores are a ranking aid and have not been calibrated that way.' },
    'sensationalism': { label: 'Sensationalism', definition: 'Share of the wording made of sensational or emotive terms, from 0 to 1. A style cue, common in false posts, but not evidence that a claim is false.' },
    'source rate': { label: 'Source fake or misleading rate', definition: 'How often this source rates the claims it covers as fake or misleading. Fact-checkers mostly publish debunks, so a high rate is normal.' },
    'cross-checker agreement': { label: 'Cross-checker agreement', definition: 'The number of distinct fact-checking sources that rated a near-identical claim. More independent sources agreeing is stronger evidence.' },
    'propaganda technique': { label: 'Propaganda technique', definition: 'One of 14 recognised persuasion patterns, such as loaded language or flag-waving, that the span encoder looks for in the wording of a post. A technique describes how something is said, not whether it is true.' },
    'signal rate': { label: 'Signal rate', definition: 'Share of posts that contain at least one strong propaganda span. Shown beside lift; it is the raw figure that lift corrects for post length.' },
    'debunk rate': { label: 'Debunk rate', definition: 'Share of posts whose nearest fact-check is rated fake or misleading and is a close match in wording and meaning. A lead for review, not a ruling on the post.' },
    'unlocated': { label: 'Unlocated', definition: 'Posts that could not be tied to a country: no country is named in the text (Mentioned view) or the channel is not in the reviewed registry (Posted from view).' },
    'narrative': { label: 'Narrative', definition: 'A group of similar posts from one country, found by clustering their meaning, and named neutrally by a language model. Shows what people are talking about, not whether it is right.' },
    'cohesion': { label: 'Cohesion', definition: 'How closely the posts in a narrative resemble each other, from 0 to 1. Higher means a tighter, more repetitive story.' },
    'volume': { label: 'Volume', definition: 'The number of posts attributed to a country in the current view. Colours use a log scale because a few countries have far more posts than the rest.' },
    'loaded language': { label: 'Loaded Language', definition: 'Words chosen for emotional weight so that readers react before they think.' },
    'name calling / labeling': { label: 'Name Calling / Labeling', definition: 'Attaching a loaded label to a person, group or idea to win approval or rejection.' },
    'repetition': { label: 'Repetition', definition: 'Repeating a message or phrase until it feels true.' },
    'exaggeration / minimisation': { label: 'Exaggeration / Minimisation', definition: 'Making something look much larger, or much smaller, than the facts support.' },
    'doubt': { label: 'Doubt', definition: 'Questioning the credibility of a person, institution or fact without evidence.' },
    'appeal to fear-prejudice': { label: 'Appeal to fear-prejudice', definition: 'Building support by stirring fear or prejudice about a group or outcome.' },
    'flag-waving': { label: 'Flag-Waving', definition: 'Playing on national or group pride to justify a position.' },
    'causal oversimplification': { label: 'Causal Oversimplification', definition: 'Blaming one cause for something that has many causes.' },
    'slogans': { label: 'Slogans', definition: 'A short, catchy phrase that packs a message into something easy to repeat.' },
    'appeal to authority': { label: 'Appeal to Authority', definition: 'Claiming something is true because an authority or expert said so, without further support.' },
    'black-and-white fallacy': { label: 'Black-and-White Fallacy', definition: 'Presenting only two options when more exist.' },
    'thought-terminating cliches': { label: 'Thought-terminating Cliches', definition: 'A stock phrase that shuts down discussion, such as "it is what it is".' },
    'whataboutism / straw men / red herring': { label: 'Whataboutism / Straw Men / Red Herring', definition: 'Deflecting criticism by pointing elsewhere, misrepresenting the other side, or changing the subject.' },
    'bandwagon / reductio ad hitlerum': { label: 'Bandwagon / Reductio ad hitlerum', definition: 'Urging people to join because everyone else has, or discrediting an idea by comparing it to something hated.' }
  };
  var GKEY = {}; Object.keys(GLOSSARY).forEach(function (k) { GKEY[k.toLowerCase()] = k; });
  function glossaryEntry(term) { var k = GKEY[String(term || '').toLowerCase()]; return k ? GLOSSARY[k] : null; }
  /* term(key [, text]) -> a focusable label with a (?) that shows the definition on hover and keyboard focus.
     The shared tooltip element is positioned by shell.js. */
  function term(key, text) {
    var g = glossaryEntry(key);
    if (!g) return esc(text || key);
    return '<span class="term" tabindex="0" data-term="' + esc(GKEY[String(key).toLowerCase()]) + '">' +
      '<span class="term-t">' + esc(text || g.label) + '</span><span class="term-q" aria-hidden="true">?</span><span class="sr"> (definition available)</span></span>';
  }

  window.Labels = {
    site: site, platform: platform, platformTable: platformTable, isSocial: isSocial, channel: channel, lang: lang, country: country, verdict: verdict, verdictPlain: verdictPlain, technique: technique, tier: tier, scoreSource: scoreSource, verdictBasis: verdictBasis, titleCase: titleCase,
    dateLong: dateLong, dateTimeUTC: dateTimeUTC, glossary: GLOSSARY, glossaryEntry: glossaryEntry, term: term, esc: esc
  };
})();
