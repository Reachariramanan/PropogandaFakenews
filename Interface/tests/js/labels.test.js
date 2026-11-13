'use strict';
/* labels.js is a browser IIFE: give it a bare `window` and read what it registers. */
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
globalThis.window = {};
require(path.join(__dirname, '..', '..', 'web', 'labels.js'));
const L = globalThis.window.Labels;

test('X and Twitter read the same, the source table stays visible', () => {
  assert.equal(L.platform('twitter'), 'X (Twitter)'); assert.equal(L.platform('twitter_posts'), 'X (Twitter)');
  assert.equal(L.platform('x'), 'X (Twitter)'); assert.equal(L.platform('x_posts'), 'X (Twitter)');
  assert.equal(L.platformTable('twitter'), 'twitter_posts'); assert.equal(L.platformTable('x'), 'x_posts');
  assert.equal(L.platformTable('x_posts'), 'x_posts'); assert.equal(L.platformTable('news'), 'scraped_articles'); assert.equal(L.platformTable(''), '');
});

test('platform names and fallbacks', () => {
  assert.equal(L.platform('reddit_posts'), 'Reddit'); assert.equal(L.platform('truthsocial_posts'), 'Truth Social'); assert.equal(L.platform(''), 'Unknown platform');
  assert.equal(L.platform('some_new_posts'), 'Some New');
});

test('esc and language names', () => {
  assert.equal(L.esc('<b>"x"</b>'), '&lt;b&gt;&quot;x&quot;&lt;/b&gt;'); assert.equal(L.lang('hi-IN'), 'Hindi'); assert.equal(L.lang('und'), 'Unknown');
});
