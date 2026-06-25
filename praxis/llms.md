# Praxis Setup

Praxis is a Codex plugin. It installs hooks that inject the Praxis prompt at chat start and before write-capable tool use on code/config files.

Plugin link: https://github.com/coder-2011/praxis

## Install

1. Clone or fork this repo.
2. From anywhere, add this repo as a Codex plugin marketplace:

```sh
codex plugin marketplace add /path/to/praxis
```

3. Install the plugin:

```sh
codex plugin add praxis@praxis
```

4. Start a new Codex thread so the hooks and bundled skills are loaded.

## Update A Local Checkout

1. Pull or edit the repo.
2. Reinstall the plugin:

```sh
codex plugin add praxis@praxis
```

3. Start a new Codex thread.

## Verify

Run the test suite:

```sh
npm test
```

Check that the installed plugin is enabled:

```sh
codex plugin list
```

The installed Praxis entry should show `installed, enabled`.
