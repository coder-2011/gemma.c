const assert = require('node:assert');
const fs = require('node:fs');
const { spawnSync } = require('node:child_process');
const test = require('node:test');
const path = require('node:path');
const { METACOGNITIVE_PROMPT } = require('../hooks/metacognitive-instructions');

const root = path.join(__dirname, '..');

function runHook(file, payload = {}, env = {}) {
  return spawnSync(process.execPath, [path.join(root, 'hooks', file)], {
    input: JSON.stringify(payload),
    encoding: 'utf8',
    env: { ...process.env, ...env },
  });
}

// Reads a skill file and returns only the loaded instruction body.
function skillBody(skillName) {
  return fs.readFileSync(path.join(root, 'skills', skillName, 'SKILL.md'), 'utf8')
    // Strip required skill frontmatter so the comparison covers the prompt text.
    .replace(/^---\n[\s\S]*?\n---\n?/, '')
    .trim();
}

test('session hook emits praxis context for Codex', () => {
  const result = runHook('metacognitive-activate.js', {}, { PLUGIN_DATA: '/tmp/plugin-data' });
  assert.equal(result.status, 0);
  const output = JSON.parse(result.stdout);
  assert.equal(output.systemMessage, 'PRAXIS');
  assert.ok(output.hookSpecificOutput.additionalContext.startsWith('This is a note I am giving you before you write code'));
  assert.match(output.hookSpecificOutput.additionalContext, /The best code is the code never written/);
  assert.match(output.hookSpecificOutput.additionalContext, /Collateral Side Effects/);
});

test('pre-tool hook ignores read-only tools', () => {
  const result = runHook('metacognitive-pre-tool.js', { tool_name: 'Read' });
  assert.equal(result.status, 0);
  assert.equal(result.stdout, '');
});

test('pre-tool hook emits before direct write tools', () => {
  const result = runHook('metacognitive-pre-tool.js', {
    tool_name: 'Write',
    tool_input: { file_path: 'src/app.ts' },
  }, { PLUGIN_DATA: '/tmp/plugin-data' });
  assert.equal(result.status, 0);
  const output = JSON.parse(result.stdout);
  assert.equal(output.hookSpecificOutput.hookEventName, 'PreToolUse');
});

test('pre-tool hook ignores non-code file edits', () => {
  const result = runHook('metacognitive-pre-tool.js', {
    tool_name: 'Write',
    tool_input: { file_path: 'README.md' },
  });
  assert.equal(result.status, 0);
  assert.equal(result.stdout, '');
});

test('pre-tool hook distinguishes shell reads from writes', () => {
  const read = runHook('metacognitive-pre-tool.js', {
    tool_name: 'Bash',
    tool_input: { command: 'sed -n "1,20p" file.js' },
  });
  assert.equal(read.stdout, '');

  const write = runHook('metacognitive-pre-tool.js', {
    tool_name: 'Bash',
    tool_input: { command: 'rm file.js' },
  }, { PLUGIN_DATA: '/tmp/plugin-data' });
  assert.match(JSON.parse(write.stdout).hookSpecificOutput.additionalContext, /write-capable tool use/);

  const docs = runHook('metacognitive-pre-tool.js', {
    tool_name: 'Bash',
    tool_input: { command: 'rm README.md' },
  });
  assert.equal(docs.stdout, '');
});

test('pre-tool hook detects code files in patches and package manager writes', () => {
  const patch = runHook('metacognitive-pre-tool.js', {
    tool_name: 'apply_patch',
    tool_input: { input: '*** Begin Patch\n*** Update File: src/main.rs\n@@\n+fn main() {}\n*** End Patch' },
  }, { PLUGIN_DATA: '/tmp/plugin-data' });
  assert.match(JSON.parse(patch.stdout).hookSpecificOutput.additionalContext, /write-capable tool use/);

  const docsPatch = runHook('metacognitive-pre-tool.js', {
    tool_name: 'apply_patch',
    tool_input: { input: '*** Begin Patch\n*** Update File: README.md\n@@\n+notes\n*** End Patch' },
  });
  assert.equal(docsPatch.stdout, '');

  const packageManager = runHook('metacognitive-pre-tool.js', {
    tool_name: 'Bash',
    tool_input: { command: 'npm install lodash' },
  }, { PLUGIN_DATA: '/tmp/plugin-data' });
  assert.match(JSON.parse(packageManager.stdout).hookSpecificOutput.additionalContext, /write-capable tool use/);
});

// Verifies the shipped audit skill stays exactly aligned with the hook prompt.
test('praxis-audit skill body matches hook prompt exactly', () => {
  assert.equal(skillBody('praxis-audit'), METACOGNITIVE_PROMPT.trim());
});
