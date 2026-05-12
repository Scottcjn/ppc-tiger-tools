# Contributing

Thanks for helping improve ppc-tiger-tools. This repository supports legacy
PowerPC Mac OS X systems, so changes should be conservative, reproducible, and
well documented.

## Local Setup

Clone the repository:

```bash
git clone https://github.com/Scottcjn/ppc-tiger-tools.git
cd ppc-tiger-tools
```

Read the existing documentation and scripts before changing behavior.

## Contribution Guidelines

- Preserve compatibility with the documented Tiger, Leopard, and PowerPC
  targets.
- Avoid introducing dependencies that are difficult to obtain on legacy systems.
- Keep install and build steps explicit.
- Update documentation when changing toolchain, compiler, or setup behavior.
- Prefer focused pull requests that change one tool, script, or document at a
  time.

## Validation

For documentation-only changes:

```bash
git diff --check
```

For script or toolchain changes, include the exact commands and target platform
used for validation. If testing on real hardware is not possible, state that
clearly in the pull request.

## Pull Request Checklist

- Summarize the affected tool or setup path.
- Include validation commands and platform details.
- Note any compatibility risks.
- Link the related issue or bounty, if applicable.
