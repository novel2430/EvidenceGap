# EvidenceGap Frontend

V0 的 React + TypeScript + Vite 前端，用於展示醫療結論的證據鏈、證據缺口與安全結論比較。

## 開發

```bash
pnpm install
pnpm dev
```

專案要求 Node.js 24，套件管理器為 pnpm 11。

## 可調整工作區

工作台使用 `react-resizable-panels` 建立巢狀可調整版面：

- 拖曳左側案例列表、中央 Claim Graph、右側 Evidence Inspector 之間的直向分隔線，可調整各欄寬度。
- 拖曳主工作區與底部報告之間的橫向分隔線，可調整上下高度。
- 拖曳 Gap Report 與 Conclusion Compare 之間的分隔線，可調整底部兩區寬度。
- 分隔線取得焦點後，可用方向鍵微調尺寸。

各面板都設定了最低尺寸，避免把內容壓縮到不可讀。
