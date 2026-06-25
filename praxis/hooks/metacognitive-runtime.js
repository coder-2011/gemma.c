const isCopilot = Boolean(process.env.COPILOT_PLUGIN_DATA);
const isCodex = !isCopilot && Boolean(process.env.PLUGIN_DATA);

function readHookInput(callback) {
  let input = '';
  process.stdin.on('data', (chunk) => { input += chunk; });
  process.stdin.on('end', () => {
    if (!input.trim()) {
      callback({});
      return;
    }
    try {
      callback(JSON.parse(input.replace(/^\uFEFF/, '')));
    } catch {
      callback({});
    }
  });
}

function writeHookOutput(event, context) {
  if (isCopilot) {
    process.stdout.write(JSON.stringify(
      event === 'SessionStart' && context ? { additionalContext: context } : {},
    ));
    return;
  }

  if (isCodex) {
    const output = { systemMessage: 'PRAXIS' };
    if (context) {
      output.hookSpecificOutput = {
        hookEventName: event,
        additionalContext: context,
      };
    }
    process.stdout.write(JSON.stringify(output));
    return;
  }

  process.stdout.write(context || '');
}

function writeNothing() {
  process.stdout.write('');
}

function toolNameFromPayload(data) {
  return String(
    data.tool_name ||
    data.toolName ||
    data.tool ||
    data.name ||
    data.invocation?.toolName ||
    '',
  );
}

function commandFromPayload(data) {
  return String(
    data.tool_input?.command ||
    data.tool_input?.cmd ||
    data.input?.command ||
    data.input?.cmd ||
    data.arguments?.command ||
    data.arguments?.cmd ||
    data.command ||
    data.cmd ||
    '',
  );
}

const WRITE_TOOL_RE = /(Write|Edit|MultiEdit|NotebookEdit|Delete|Remove|apply_patch)/i;
const SHELL_TOOL_RE = /(Bash|Shell|exec_command|write_stdin)/i;
const WRITE_COMMAND_RE = /\b(apply_patch|rm|rmdir|mv|cp|mkdir|touch|chmod|chown|tee|sed\s+-i|perl\s+-pi|npm\s+(install|i|add)|bun\s+add|pnpm\s+add|yarn\s+add)\b|>\s*[^&|]|>>\s*[^&|]/i;
const PACKAGE_MANAGER_RE = /\b(npm\s+(install|i|add)|bun\s+add|pnpm\s+add|yarn\s+add)\b/i;
const CODE_EXTENSIONS = new Set([
  '.ada', '.adb', '.ads', '.ahk', '.apex', '.astro', '.awk', '.bash', '.bat',
  '.bazel', '.bicep', '.bzl', '.c', '.cairo', '.capnp', '.cc', '.circom',
  '.cjs', '.clj', '.cljs', '.cljc', '.cmake', '.cmd', '.coffee', '.cpp',
  '.cs', '.csh', '.css', '.cts', '.cu', '.cuh', '.cxx', '.dart', '.dhall',
  '.dts', '.eex', '.ejs', '.elm', '.erb', '.erl', '.ex', '.exs', '.fish',
  '.fs', '.fsi', '.fsx', '.gemspec', '.gitattributes', '.gitignore', '.glsl',
  '.go', '.gql', '.gradle', '.graphql', '.groovy', '.h', '.haml', '.handlebars',
  '.hcl', '.hh', '.hpp', '.hrl', '.hs', '.htm', '.html', '.hxx', '.ini',
  '.ino', '.ipynb', '.java', '.jinja', '.jinja2', '.jl', '.js', '.json',
  '.json5', '.jsonc', '.jsx', '.kt', '.kts', '.less', '.lhs', '.liquid',
  '.lua', '.m', '.make', '.mdx', '.mjs', '.mk', '.ml', '.mli', '.mm', '.move',
  '.mts', '.mustache', '.nim', '.nix', '.nomad', '.odin', '.php', '.phtml',
  '.pl', '.plist', '.pm', '.postcss', '.prisma', '.proto', '.ps1', '.psd1',
  '.psm1', '.pug', '.py', '.pyi', '.pyw', '.q', '.qml', '.r', '.rake', '.rb',
  '.rei', '.rego', '.re', '.rkt', '.rmd', '.rs', '.rq', '.sass', '.scala',
  '.sc', '.scss', '.sed', '.sh', '.slim', '.sln', '.sol', '.sparql', '.sql',
  '.styl', '.svelte', '.sv', '.svg', '.svh', '.swift', '.t', '.tf',
  '.tfvars', '.thrift', '.toml', '.ts', '.tsx', '.twig', '.v', '.vb',
  '.vbs', '.vcxproj', '.vh', '.vhd', '.vhdl', '.vue', '.vy', '.wxml',
  '.wxss', '.xaml', '.xib', '.xml', '.xsd', '.xsl', '.xslt', '.yaml', '.yml',
  '.zig', '.zsh',
]);
const CODE_FILENAMES = new Set([
  '.babelrc', '.browserslistrc', '.clang-format', '.clang-tidy',
  '.dockerignore', '.editorconfig', '.env', '.env.example', '.env.local',
  '.eslintrc', '.gitattributes', '.gitignore', '.npmrc', '.nvmrc',
  '.prettierrc', '.python-version', '.ruby-version', '.tool-versions',
  'Appfile', 'Brewfile', 'BUCK', 'BUILD', 'bun.lock', 'bun.lockb',
  'Cargo.lock', 'Cargo.toml', 'CMakeLists.txt', 'Containerfile',
  'deno.json', 'deno.lock', 'Deliverfile', 'Dockerfile', 'Earthfile',
  'Fastfile', 'Gemfile', 'Gemfile.lock', 'go.mod', 'go.sum', 'Guardfile',
  'Jenkinsfile', 'Justfile', 'Makefile', 'MODULE.bazel', 'Matchfile',
  'package-lock.json', 'package.json', 'Pipfile', 'Pipfile.lock', 'pnpm-lock.yaml',
  'Podfile', 'poetry.lock', 'Procfile', 'pyproject.toml', 'Rakefile',
  'requirements.txt', 'Scanfile', 'setup.cfg', 'Snapfile', 'Taskfile',
  'Tiltfile', 'tsconfig.json', 'Vagrantfile', 'WORKSPACE', 'yarn.lock',
]);
const PATH_TOKEN_RE = /(?:\.{0,2}\/|~\/)?[A-Za-z0-9_@+.,%=-][A-Za-z0-9_@+.,%=:\/\\-]*(?:\.[A-Za-z0-9][A-Za-z0-9.+-]*|\b(?:Dockerfile|Containerfile|Makefile|Justfile|Taskfile|Procfile|Brewfile|Gemfile|Rakefile|Guardfile|Podfile|Jenkinsfile|Tiltfile|Earthfile|BUILD|WORKSPACE)\b)/g;

function shouldInjectForTool(data) {
  const toolName = toolNameFromPayload(data);
  if (!toolName) return hasCodePath(data);
  if (WRITE_TOOL_RE.test(toolName)) return hasCodePath(data);
  if (!SHELL_TOOL_RE.test(toolName)) return false;

  const command = commandFromPayload(data);
  if (!command || !WRITE_COMMAND_RE.test(command)) return false;
  return PACKAGE_MANAGER_RE.test(command) || hasCodePath(data);
}

function hasCodePath(data) {
  return pathCandidatesFromPayload(data).some(isCodePath);
}

function pathCandidatesFromPayload(data) {
  const candidates = [];
  collectStrings(data, candidates, 0);
  return candidates.flatMap(extractPaths);
}

function collectStrings(value, candidates, depth) {
  if (depth > 5 || value == null) return;
  if (typeof value === 'string') {
    candidates.push(value);
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item) => collectStrings(item, candidates, depth + 1));
    return;
  }
  if (typeof value === 'object') {
    Object.values(value).forEach((item) => collectStrings(item, candidates, depth + 1));
  }
}

function extractPaths(text) {
  const paths = [];
  for (const line of String(text).split(/\r?\n/)) {
    const patchPath = line.match(/^(?:\*\*\* (?:Add|Update|Delete) File:|\*\*\* Move to:|[+-]{3} [ab]\/|diff --git a\/\S+ b\/)(?:\s*)?(.+)$/);
    if (patchPath) paths.push(...patchPath[1].trim().split(/\s+b\//).slice(-1));
    paths.push(...(line.match(PATH_TOKEN_RE) || []));
  }
  return paths;
}

function isCodePath(rawPath) {
  const normalized = String(rawPath)
    .trim()
    .replace(/^["'`]+|["'`,:;)\]]+$/g, '')
    .replace(/\\/g, '/')
    .replace(/^[ab]\//, '');
  const basename = normalized.split('/').pop();
  if (!basename) return false;
  if (CODE_FILENAMES.has(basename)) return true;
  const dot = basename.lastIndexOf('.');
  if (dot <= 0) return false;
  return CODE_EXTENSIONS.has(basename.slice(dot).toLowerCase());
}

module.exports = {
  commandFromPayload,
  hasCodePath,
  isCodePath,
  pathCandidatesFromPayload,
  readHookInput,
  shouldInjectForTool,
  toolNameFromPayload,
  writeHookOutput,
  writeNothing,
};
