# GridSynergy - 初始化Git并推送到GitHub
# 用法: 右键 -> 使用PowerShell运行
# 需要先安装Git: https://git-scm.com/download/win

$repo = "git@github.com:Ze-fan1/-GridSynergy.git"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GridSynergy GitHub 初始化脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查Git
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    Write-Host "[错误] 未找到 Git，请先安装: https://git-scm.com/download/win" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

Write-Host "[1/6] 初始化 Git 仓库..." -ForegroundColor Yellow
Set-Location $PSScriptRoot
git init

Write-Host "[2/6] 添加远程仓库..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin $repo

Write-Host "[3/6] 添加所有文件..." -ForegroundColor Yellow
git add .
git add frontend/src/ -f

Write-Host "[4/6] 提交..." -ForegroundColor Yellow
git commit -m "v1: GridSynergy新能源电网自主调度系统
- React 19 + Vite 前端 (Framer Motion, Chart.js, React Router)
- Python后端 (FastAPI, LLM Agent, SSE流式响应)
- 12个预设场景 + 自定义调度
- 多Agent协作日志 + 多方法对比实验
- 雷达图 + 消融实验 + 调度历史记录"

Write-Host "[5/6] 推送到 GitHub..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[注意] 推送失败，请检查:" -ForegroundColor Red
    Write-Host "  1. SSH密钥是否已添加到 GitHub: https://github.com/settings/keys" -ForegroundColor Gray
    Write-Host "  2. 仓库地址是否可访问: $repo" -ForegroundColor Gray
    Write-Host ""
    Write-Host "如果使用 HTTPS，改为:" -ForegroundColor Yellow
    Write-Host "  git remote set-url origin https://github.com/Ze-fan1/-GridSynergy.git" -ForegroundColor Gray
    Write-Host "  git push -u origin main" -ForegroundColor Gray
    Read-Host "按回车键退出"
    exit 1
}

Write-Host "[6/6] 打标签 v1..." -ForegroundColor Yellow
git tag -a v1 -m "v1: GridSynergy 第一版"
git push origin v1

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  推送成功!" -ForegroundColor Green
Write-Host "  仓库地址: https://github.com/Ze-fan1/-GridSynergy" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Read-Host "按回车键退出"
