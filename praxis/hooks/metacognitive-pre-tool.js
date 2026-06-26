#!/usr/bin/env node

const { getInstructions } = require('./metacognitive-instructions');
const {
  readHookInput,
  shouldInjectForTool,
  writeHookOutput,
  writeNothing,
} = require('./metacognitive-runtime');

readHookInput((data) => {
  if (!shouldInjectForTool(data)) {
    writeNothing();
    return;
  }
  writeHookOutput('PreToolUse', getInstructions('write-capable tool use'));
});
