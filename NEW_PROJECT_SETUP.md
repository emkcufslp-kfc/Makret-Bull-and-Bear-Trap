# New Project Setup

建立新 ChatGPT、Codex 或本機專案時，將下列兩個檔案複製到專案根目錄：

- `PROJECT_OPERATIONAL_PROTOCOL.md`：完整工作規範。
- `SYSTEM_PROMPT.md`：可貼入 ChatGPT/Codex 專案 instructions 或 system prompt 的精簡版。

建議在專案的 `AGENTS.md` 或專案說明中加入：

```text
請遵循同目錄的 PROJECT_OPERATIONAL_PROTOCOL.md 與 SYSTEM_PROMPT.md。
```

不要覆蓋專案既有的 system prompt、AGENTS.md 或使用者設定；如有衝突，以較高優先級的系統/開發者指令為準。
