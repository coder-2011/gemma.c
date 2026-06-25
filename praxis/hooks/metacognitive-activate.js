#!/usr/bin/env node

const { getInstructions } = require('./metacognitive-instructions');
const { writeHookOutput } = require('./metacognitive-runtime');

writeHookOutput('SessionStart', getInstructions('session start'));
