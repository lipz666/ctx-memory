#!/usr/bin/env bash
# Install the committed ctx source as this machine's ctx daemon, reversibly.
#
#   tools/install-local.sh              build HEAD and install it
#   tools/install-local.sh --rollback   restore the binary from the latest backup
#
# Install: refuses uncommitted changes to build inputs; builds; backs up the current
# binary, config, memory files and state database; swaps the binary atomically; restarts
# the launchd service; checks that the daemon answers with the new commit, and rolls back
# automatically if it does not; then backfills vectors and conversation excerpts
# (`ctx reindex`). Every install is logged in $CTX_HOME/install-log.jsonl and the
# installed commit is tagged `installed` in this repository.
#
# CTX_HOME selects another instance (default ~/.ctx); without a launchd service for it,
# the restart and health check are skipped.
set -euo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
HOME_DIR=${CTX_HOME:-$HOME/.ctx}
BIN="$HOME_DIR/bin/ctx"
LABEL=dev.ctx.runtime
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
# The default instance keeps its event key in the Keychain only when CTX_HOME is unset,
# so child commands must not see an explicit CTX_HOME for it.
run_ctx() { if [ -n "${CTX_HOME:-}" ]; then CTX_HOME="$CTX_HOME" "$BIN" "$@"; else env -u CTX_HOME "$BIN" "$@"; fi; }
managed() { [ -z "${CTX_HOME:-}" ] && [ -f "$PLIST" ]; }

restart() {
  if managed; then
    launchctl kickstart -k "gui/$(id -u)/$LABEL"
  else
    echo "no launchd service for $HOME_DIR; restart the daemon yourself"
  fi
}

# Wait until the daemon answers /api/v1/health; print its version.
health() {
  local port token
  port=$(sed -n 's/^port: *\([0-9]*\).*/\1/p' "$HOME_DIR/config.yaml")
  token=$(cat "$HOME_DIR/token")
  for _ in $(seq 60); do
    if out=$(curl -sf --max-time 2 -H "X-Ctx-Token: $token" "http://127.0.0.1:$port/api/v1/health"); then
      echo "$out" | sed -n 's/.*"version":"\([^"]*\)".*/\1/p'
      return 0
    fi
    sleep 1
  done
  return 1
}

latest_backup() { ls -1d "$HOME_DIR"/backups/install-* 2>/dev/null | tail -1; }

if [ "${1:-}" = "--rollback" ]; then
  backup=$(latest_backup)
  [ -n "$backup" ] || { echo "no install backup in $HOME_DIR/backups" >&2; exit 1; }
  cp "$backup/ctx" "$BIN.new" && mv -f "$BIN.new" "$BIN"
  restart
  managed && echo "daemon: $(health || echo 'not answering')"
  echo "restored $("$BIN" --version) from $backup"
  exit 0
fi

cd "$ROOT"
if ! git diff --quiet HEAD -- src Cargo.toml Cargo.lock build.rs prompts web \
  || [ -n "$(git ls-files --others --exclude-standard -- src prompts web)" ]; then
  echo "uncommitted changes in build inputs; commit them first so the install maps to a commit" >&2
  exit 1
fi
commit=$(git rev-parse --short HEAD)
cargo build --release --quiet
new_version=$(target/release/ctx --version)
case "$new_version" in *"($commit)"*) ;; *) echo "built binary reports '$new_version', expected $commit" >&2; exit 1 ;; esac

old_version=$("$BIN" --version 2>/dev/null || echo "ctx unknown")
backup="$HOME_DIR/backups/install-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup"
cp -p "$BIN" "$backup/ctx"
cp -p "$HOME_DIR/config.yaml" "$backup/"
cp -Rp "$HOME_DIR/memory" "$backup/memory"
if [ -f "$HOME_DIR/state/events.db" ]; then
  mkdir -p "$backup/state"
  # A consistent copy of the live database (WAL included).
  sqlite3 "$HOME_DIR/state/events.db" ".backup '$backup/state/events.db'"
fi
echo "backup: $backup ($old_version)"

cp target/release/ctx "$BIN.new" && mv -f "$BIN.new" "$BIN"
restart
if managed; then
  running=$(health || true)
  case "$running" in
    *"($commit)"*) echo "daemon: $running" ;;
    *)
      echo "daemon did not come back with $commit (got '${running:-no answer}'); rolling back" >&2
      cp "$backup/ctx" "$BIN.new" && mv -f "$BIN.new" "$BIN"
      restart
      echo "rolled back to $(health || echo 'not answering')" >&2
      exit 1
      ;;
  esac
fi

run_ctx reindex || echo "reindex failed; run 'ctx reindex' later" >&2
printf '{"at":"%s","from":"%s","to":"%s","backup":"%s"}\n' "$(date -u +%FT%TZ)" "$old_version" "$new_version" "$backup" \
  >> "$HOME_DIR/install-log.jsonl"
[ -z "${CTX_HOME:-}" ] && git tag -f installed HEAD >/dev/null
echo "installed $new_version"
