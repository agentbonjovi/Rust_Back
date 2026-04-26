#!/usr/bin/env python3
"""Надежный GUI без tkinter: локальный web-интерфейс на стандартной библиотеке."""

from __future__ import annotations

import argparse
import cgi
import html
import json
import csv
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT_DIR / "synthetic_data.csv"
REPORT_FILE = ROOT_DIR / "ml_optimizer_knn_report.txt"
LOCAL_DASHBOARD_FILE = ROOT_DIR / "ml_optimizer_dashboard.html"
UPLOAD_DIR = ROOT_DIR / ".uploads"
HOST = "127.0.0.1"
START_PORT = 8765
MAX_ROWS_FOR_PREVIEW = 5000


HTML_PAGE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ML Optimizer</title>
  <style>
    :root {
      --bg: #eef2f7;
      --bg-2: #e6ecf4;
      --panel: #ffffff;
      --panel-soft: #f8fafc;
      --ink: #0f172a;
      --muted: #64748b;
      --accent: #6366f1;          /* indigo  */
      --accent-soft: #eef0ff;
      --accent-2: #06b6d4;         /* cyan    */
      --accent-3: #10b981;         /* emerald */
      --accent-4: #f59e0b;         /* amber   */
      --accent-5: #8b5cf6;         /* violet  */
      --accent-6: #ef4444;         /* red     */
      --border: #dbe2ec;
      --border-strong: #c7d0dc;
      --shadow-sm: 0 1px 2px rgba(15, 23, 42, .04), 0 1px 3px rgba(15, 23, 42, .06);
      --shadow-md: 0 4px 12px rgba(15, 23, 42, .06), 0 10px 28px rgba(15, 23, 42, .08);
      --shadow-lg: 0 12px 40px rgba(99, 102, 241, .18);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Inter", "SF Pro Text", "Segoe UI", system-ui, sans-serif;
      background-color: var(--bg);
      /* лёгкая точечная сетка + два мягких градиентных пятна сверху */
      background-image:
        radial-gradient(circle at 12% 8%, rgba(99, 102, 241, .14), transparent 38%),
        radial-gradient(circle at 92% 6%, rgba(6, 182, 212, .12), transparent 36%),
        radial-gradient(rgba(15, 23, 42, .055) 1px, transparent 1px);
      background-size: 100% 100%, 100% 100%, 22px 22px;
      background-position: 0 0, 0 0, 0 0;
      background-attachment: fixed;
      color: var(--ink);
    }
    .wrap {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 18px 36px;
    }
    .hero {
      margin-bottom: 18px;
      padding: 22px 26px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: linear-gradient(135deg, rgba(255, 255, 255, .94) 0%, rgba(238, 240, 255, .9) 100%);
      backdrop-filter: blur(8px);
      box-shadow: var(--shadow-md);
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4.4vw, 40px);
      line-height: 1.05;
      letter-spacing: -0.02em;
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 18px;
    }
    .charts {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    @media (max-width: 1100px) {
      .charts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 600px) {
      .charts { grid-template-columns: 1fr; }
    }
    .stats {
      display: grid;
      /* 6 тайлов в одну строку; на узких экранах сворачивается в 3, потом в 2 */
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-top: 16px;
    }
    .stat {
      position: relative;
      padding: 12px 14px 10px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: linear-gradient(180deg, #ffffff 0%, var(--panel-soft) 100%);
      transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
      cursor: default;
      overflow: hidden;
      isolation: isolate;
    }
    /* цветная полоска сверху — у каждого тайла свой акцент */
    .stat::before {
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 3px;
      background: var(--tile-color, var(--accent));
      opacity: .8;
      transition: opacity .18s ease, height .18s ease;
    }
    .stat:hover {
      transform: translateY(-3px);
      box-shadow: var(--shadow-lg);
      border-color: var(--tile-color, var(--accent));
      background: linear-gradient(180deg, #ffffff 0%, var(--accent-soft) 100%);
    }
    .stat:hover::before { height: 4px; opacity: 1; }
    .stat .label {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .04em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .stat strong {
      display: block;
      font-size: 22px;
      line-height: 1.15;
      margin-top: 4px;
      font-variant-numeric: tabular-nums;
      letter-spacing: -0.01em;
      color: var(--ink);
    }
    .stat .hint {
      color: var(--muted);
      font-size: 11px;
      margin-top: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    @media (max-width: 1100px) {
      .stats { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (max-width: 700px) {
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow-sm);
      transition: box-shadow .18s ease, border-color .18s ease;
    }
    .card:hover { box-shadow: var(--shadow-md); }
    label {
      display: block;
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 8px;
    }
    input[type=text] {
      width: 100%;
      padding: 14px 16px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: #fff;
      font-size: 15px;
    }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      cursor: pointer;
      font-weight: 700;
      font-size: 14px;
      transition: transform .12s ease, opacity .12s ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: .55; cursor: default; transform: none; }
    .primary {
      background: linear-gradient(135deg, var(--accent), var(--accent-5));
      color: #fff;
      box-shadow: 0 4px 12px rgba(99, 102, 241, .25);
    }
    .primary:hover { box-shadow: 0 6px 18px rgba(99, 102, 241, .35); }
    .secondary {
      background: #fff;
      color: var(--ink);
      border: 1px solid var(--border);
    }
    .secondary:hover { background: var(--accent-soft); border-color: var(--accent); }
    .status {
      margin-top: 16px;
      padding: 14px 16px;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--accent-soft), #fff);
      border: 1px solid var(--border);
      color: var(--ink);
      font-size: 14px;
      min-height: 52px;
    }
    .meta {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .panel-title {
      margin: 0 0 12px;
      font-size: 16px;
    }
    pre {
      margin: 0;
      min-height: 320px;
      max-height: min(70vh, 720px);
      overflow: auto;          /* и горизонтальный, и вертикальный скролл */
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid #d9d9d9;
      background: #1e1e1e;
      color: #f8fafc;
      font: 13px/1.45 "SFMono-Regular", Menlo, "Cascadia Mono", Consolas, monospace;
      /* Не ломаем строки — таблицы ASCII должны выравниваться. Если строка
         длиннее окна, появляется горизонтальный скролл вместо переносов. */
      white-space: pre;
      tab-size: 2;
    }
    .report {
      background: var(--panel-soft);
      color: var(--ink);
      border-color: var(--border);
    }
    /* "Вывод процесса" и "Отчёт" — каждый во всю ширину, друг под другом.
       Текст моноширинный и широкий, в две колонки выглядит сжатым. */
    .stack {
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }
    /* Свёртываемые панели через <details> */
    details.card { padding: 0; }
    details.card > summary {
      list-style: none;
      cursor: pointer;
      padding: 18px 18px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    details.card > summary::-webkit-details-marker { display: none; }
    details.card > summary .panel-title { margin: 0; flex: 1; }
    details.card > summary .chev {
      transition: transform .15s ease;
      color: var(--muted);
      font-size: 14px;
      user-select: none;
    }
    details.card[open] > summary .chev { transform: rotate(90deg); }
    details.card > .panel-body {
      padding: 0 18px 18px;
    }
    /* Тулбар с кнопками "Копировать" / "Скачать" */
    .panel-tools {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .tool-btn {
      border: 1px solid var(--border);
      background: #fff;
      color: var(--ink);
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: background .12s ease, transform .12s ease, border-color .12s ease, color .12s ease;
    }
    .tool-btn:hover {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--accent);
      transform: translateY(-1px);
    }
    .tool-btn.copied {
      background: #d1fae5;
      border-color: var(--accent-3);
      color: #065f46;
    }
    /* Подсветка строк во всех таблицах */
    tbody tr { transition: background .12s ease; }
    tbody tr:hover { background: var(--accent-soft); }
    /* Таблица "Приоритеты признаков KNN" */
    .fi-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .fi-table th, .fi-table td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      white-space: nowrap;
    }
    .fi-table th {
      background: var(--panel-soft);
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
      color: var(--muted);
    }
    .fi-table code {
      background: var(--accent-soft);
      color: var(--accent);
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 12.5px;
      font-weight: 600;
    }
    .fi-table tr.fi-top td { font-weight: 700; }
    .fi-table tr.fi-top td:first-child {
      position: relative;
    }
    .fi-table tr.fi-top td:first-child::before {
      content: "★";
      position: absolute;
      left: -2px;
      color: var(--accent);
    }
    .fi-desc {
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
      white-space: normal;
      max-width: 420px;
    }
    .fi-table th { white-space: normal; }
    .fi-table .th-hint {
      display: inline-block;
      color: var(--muted);
      font-weight: 500;
      text-transform: none;
      letter-spacing: 0;
      font-size: 11px;
      margin-left: 4px;
    }
    .fi-legend {
      margin: 0 0 14px;
      padding: 14px 16px;
      background: linear-gradient(135deg, var(--accent-soft), #fff);
      border: 1px solid var(--border);
      border-left: 3px solid var(--accent);
      border-radius: 12px;
      font-size: 13px;
      color: var(--ink);
      line-height: 1.55;
    }
    .fi-legend strong { color: var(--accent); }
    .fi-legend code {
      background: #fff;
      border: 1px solid var(--border);
      padding: 1px 6px;
      border-radius: 5px;
      font-size: 12.5px;
    }
    .fi-bar-cell { width: 30%; padding-right: 14px !important; }
    .fi-bar-track {
      width: 100%;
      height: 10px;
      background: var(--bg-2);
      border-radius: 999px;
      overflow: hidden;
    }
    .fi-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      border-radius: 999px;
      transition: width .4s ease;
      box-shadow: 0 1px 4px rgba(99, 102, 241, .35);
    }
    .fi-empty { color: var(--muted); font-size: 13px; padding: 8px 0; }
    .hint {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .chart-box {
      min-height: 320px;
    }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: #fff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      background: var(--panel-soft);
      color: var(--muted);
      font-size: 12px;
      letter-spacing: .03em;
      text-transform: uppercase;
      z-index: 1;
    }
    .chart-box svg {
      width: 100%;
      height: 220px;
      display: block;
      overflow: visible;
    }
    .bar-label {
      font-size: 11px;
      fill: #64748b;
    }
    .bar-value {
      font-size: 11px;
      fill: #0f172a;
      font-weight: 700;
    }
    .chart-caption {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .empty-chart {
      min-height: 220px;
      display: grid;
      place-items: center;
      color: var(--muted);
      border: 1px dashed var(--border);
      border-radius: 14px;
      background: #fff;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      .charts { grid-template-columns: 1fr; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>ML Optimizer</h1>
    </section>

    <section class="card" style="margin-bottom: 18px;">
      <label for="dataset">CSV-файл для анализа</label>
      <input id="dataset" type="text" value="__DEFAULT_DATASET__" spellcheck="false">
      <div class="row" style="align-items: center;">
        <label style="margin: 0; font-weight: 600;">Backend:</label>
        <label style="display: inline-flex; align-items: center; gap: 6px; font-weight: 500; margin: 0;">
          <input type="radio" name="backend" value="rust" checked> Rust
        </label>
        <label style="display: inline-flex; align-items: center; gap: 6px; font-weight: 500; margin: 0;">
          <input type="radio" name="backend" value="python"> Python
        </label>
      </div>
      <div class="row">
        <button class="primary" id="runBtn">Запустить анализ</button>
        <button class="primary" id="compareBtn" style="background: var(--accent-3);">Сравнить Rust vs Python</button>
        <button class="secondary" id="generateBtn">Сгенерировать CSV</button>
        <button class="secondary" id="defaultBtn">Подставить synthetic_data.csv</button>
        <button class="secondary" id="refreshBtn">Обновить данные</button>
      </div>
      <div class="status" id="status">Загрузка состояния...</div>
      <!-- modeHint используется JS только в standalone-режиме; в browser-mode остаётся пустым -->
      <div class="hint" id="modeHint" hidden></div>
      <div class="stats" id="stats"></div>
    </section>

    <section class="charts">
      <div class="card chart-box">
        <h2 class="panel-title">Средняя ошибка моделей</h2>
        <div id="errorChart"></div>
        <div class="chart-caption" id="errorCaption"></div>
      </div>
      <div class="card chart-box">
        <h2 class="panel-title">Средняя селективность</h2>
        <div id="selectivityChart"></div>
        <div class="chart-caption" id="selectivityCaption"></div>
      </div>
      <div class="card chart-box">
        <h2 class="panel-title">Рекомендованные сканы</h2>
        <div id="scanChart"></div>
        <div class="chart-caption" id="scanCaption"></div>
      </div>
      <div class="card chart-box">
        <h2 class="panel-title">Время выполнения: Rust vs Python</h2>
        <div id="timingChart"></div>
        <div class="chart-caption" id="timingCaption"></div>
      </div>
    </section>

    <section class="card" style="margin-bottom: 18px;">
      <h2 class="panel-title">Предпросмотр CSV</h2>
      <div id="previewMeta" class="meta">Загрузка предпросмотра...</div>
      <div class="table-wrap" style="margin-top: 12px;">
        <table>
          <thead id="previewHead"></thead>
          <tbody id="previewBody"></tbody>
        </table>
      </div>
    </section>

    <section class="card" id="featureCard" style="margin-bottom: 18px;">
      <details open>
        <summary>
          <h2 class="panel-title">Приоритеты признаков KNN</h2>
          <span class="chev">▶</span>
        </summary>
        <div class="panel-body">
          <div class="fi-legend">
            <p style="margin:0 0 6px;">
              <strong>Что такое Std?</strong> &nbsp;
              <code>Std</code> (standard deviation, стандартное отклонение) — мера разброса значений признака.
              Маленький Std значит, что признак почти всегда одинаков и плохо отличает один предикат от другого;
              большой Std — наоборот, признак сильно меняется и хорошо разделяет данные.
            </p>
            <p style="margin:0;">
              <strong>Почему Std = «вес»?</strong> &nbsp;
              KNN использует <em>евклидово расстояние</em> между векторами признаков. Признак с бóльшим Std
              сильнее влияет на это расстояние, то есть сильнее «голосует» при поиске ближайших соседей.
              Все 12 признаков сначала нормализуются к диапазону [0, 1], чтобы зарплата в миллионах не
              перевесила флаг <code>operator_eq</code> ∈ {0, 1}. Колонка <em>Важность</em> — доля Std признака
              в общей сумме Std. Топ-3 отмечены ★.
            </p>
          </div>
          <div id="featureImportance"><p class="fi-empty">Запустите анализ, чтобы увидеть приоритеты признаков.</p></div>
        </div>
      </details>
    </section>

    <section class="stack">
      <details class="card" open>
        <summary>
          <h2 class="panel-title">Вывод процесса</h2>
          <div class="panel-tools">
            <button type="button" class="tool-btn" data-copy="output">Копировать</button>
            <button type="button" class="tool-btn" data-download="output" data-filename="ml_optimizer_output.txt">Скачать .txt</button>
            <span class="chev">▶</span>
          </div>
        </summary>
        <div class="panel-body">
          <pre id="output">Ожидание запуска...</pre>
        </div>
      </details>
      <details class="card" open>
        <summary>
          <h2 class="panel-title">Отчет</h2>
          <div class="panel-tools">
            <button type="button" class="tool-btn" data-copy="report">Копировать</button>
            <button type="button" class="tool-btn" data-download="report" data-filename="ml_optimizer_knn_report.txt">Скачать .txt</button>
            <span class="chev">▶</span>
          </div>
        </summary>
        <div class="panel-body">
          <pre class="report" id="report">Отчет будет показан здесь.</pre>
        </div>
      </details>
    </section>
  </div>

  <script>
    const datasetInput = document.getElementById("dataset");
    const statusEl = document.getElementById("status");
    const outputEl = document.getElementById("output");
    const reportEl = document.getElementById("report");
    const statsEl = document.getElementById("stats");
    const previewMetaEl = document.getElementById("previewMeta");
    const previewHeadEl = document.getElementById("previewHead");
    const previewBodyEl = document.getElementById("previewBody");
    const filePicker = document.getElementById("filePicker");
    const runBtn = document.getElementById("runBtn");
    const compareBtn = document.getElementById("compareBtn");
    const generateBtn = document.getElementById("generateBtn");
    const defaultBtn = document.getElementById("defaultBtn");
    const refreshBtn = document.getElementById("refreshBtn");
    function selectedBackend() {
      const el = document.querySelector("input[name=backend]:checked");
      return el ? el.value : "rust";
    }
    const defaultDataset = __DEFAULT_DATASET_JSON__;
    const appMode = "__APP_MODE__";
    const initialState = __INITIAL_STATE_JSON__;

    function esc(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function numberFmt(value) {
      return new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 }).format(value);
    }

    function renderStats(summary) {
      if (!summary) {
        statsEl.innerHTML = "";
        return;
      }
      const rustMs = summary.rust_ms;
      const pythonMs = summary.python_ms;
      const speedup = (rustMs && pythonMs) ? (pythonMs / rustMs) : null;
      // Каждому тайлу — свой акцент-цвет из CSS-переменных
      const items = [
        { label: "Столбцов",        value: numberFmt(summary.numeric_columns || 0), hint: "числовых в CSV",  color: "var(--accent)" },
        { label: "Предикатов",      value: numberFmt(summary.predicate_count || 0), hint: "проверено",       color: "var(--accent-2)" },
        { label: "Ошибка KNN",      value: numberFmt(summary.avg_error_knn || 0),   hint: "средняя",         color: "var(--accent-3)" },
        { label: "Rust, мс",        value: rustMs ? numberFmt(rustMs) : "—",        hint: "последний запуск", color: "var(--accent-4)" },
        { label: "Python, мс",      value: pythonMs ? numberFmt(pythonMs) : "—",    hint: "последний запуск", color: "var(--accent-5)" },
        { label: "Ускорение Rust",  value: speedup ? "×" + speedup.toFixed(2) : "—", hint: "Python / Rust",   color: "var(--accent-6)" },
      ];
      statsEl.innerHTML = items.map(it => `
        <div class="stat" style="--tile-color: ${it.color}" title="${esc(it.label)} — ${esc(it.hint)}">
          <span class="label">${esc(it.label)}</span>
          <strong>${esc(it.value)}</strong>
          <span class="hint">${esc(it.hint)}</span>
        </div>
      `).join("");
    }

    function renderHistogram(targetId, captionId, bins, title, options = {}) {
      const host = document.getElementById(targetId);
      const caption = document.getElementById(captionId);
      if (!bins || !bins.length) {
        host.innerHTML = '<div class="empty-chart">Недостаточно данных</div>';
        caption.textContent = "";
        return;
      }
      const width = 320;
      const height = 220;
      const transform = options.scale === "sqrt" ? Math.sqrt : (value) => value;
      const transformed = bins.map(bin => transform(bin.count));
      const max = Math.max(...transformed, 1);
      const gap = 8;
      const barWidth = (width - gap * (bins.length - 1)) / bins.length;
      const bars = bins.map((bin, index) => {
        const x = index * (barWidth + gap);
        const barHeight = (transformed[index] / max) * 170;
        const y = height - barHeight - 24;
        return `
          <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="8" fill="#0f766e" opacity="${0.55 + (index / bins.length) * 0.35}"></rect>
          <text x="${x + barWidth / 2}" y="${height - 8}" text-anchor="middle" font-size="10" fill="#6b7280">${esc(bin.label)}</text>
        `;
      }).join("");
      const axis = options.axisMax ? `
        <text x="0" y="12" font-size="10" fill="#6b7280">${esc(numberFmt(options.axisMin || 0))}</text>
        <text x="${width / 2}" y="12" text-anchor="middle" font-size="10" fill="#6b7280">${esc(numberFmt((options.axisMax || 0) / 2))}</text>
        <text x="${width}" y="12" text-anchor="end" font-size="10" fill="#6b7280">${esc(numberFmt(options.axisMax))}</text>
      ` : "";
      host.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)}">
          ${axis}
          <line x1="0" y1="${height - 24}" x2="${width}" y2="${height - 24}" stroke="#d6c7b6" />
          ${bars}
        </svg>
      `;
      const peak = bins.reduce((best, bin) => bin.count > best.count ? bin : best, bins[0]);
      const scaleHint = options.scale === "sqrt" ? "Высота столбцов сглажена, чтобы длинный хвост не ломал график." : "";
      caption.textContent = `Диапазон: ${options.axisMin ?? "auto"}-${options.axisMax ? numberFmt(options.axisMax) : "auto"}. Пиковый интервал: ${peak.label}, записей: ${numberFmt(peak.count)}. ${scaleHint}`.trim();
    }

    function colorForLabel(label, index) {
      const normalized = String(label).toLowerCase();
      if (normalized.includes("index")) return "var(--accent)";
      if (normalized.includes("seq")) return "var(--accent-4)";
      if (normalized.includes("bitmap")) return "var(--accent-5)";
      const palette = ["var(--accent)", "var(--accent-3)", "var(--accent-4)", "var(--accent-5)", "var(--accent-6)"];
      return palette[index % palette.length];
    }

    function formatChartValue(label, value) {
      const normalized = String(label).toLowerCase();
      if (normalized.includes("scan")) return String(value);
      return Number(value).toFixed(3);
    }

    function renderBarChart(targetId, captionId, items, options = {}) {
      const host = document.getElementById(targetId);
      const caption = document.getElementById(captionId);
      if (!items || !items.length) {
        host.innerHTML = '<div class="empty-chart">Недостаточно данных</div>';
        caption.textContent = "";
        return;
      }
      const width = 340;
      const leftPad = 88;
      const rightPad = 42;
      const rowHeight = 34;
      const height = items.length * rowHeight + 16;
      const max = Math.max(...items.map(item => Number(item.count) || 0), 1);
      const rows = items.map((item, index) => {
        const y = index * rowHeight + 8;
        const lineWidth = ((Number(item.count) || 0) / max) * (width - leftPad - rightPad);
        const fill = colorForLabel(item.label, index);
        return `
          <text class="bar-label" x="0" y="${y + 15}">${esc(item.label)}</text>
          <rect x="${leftPad}" y="${y}" width="${width - leftPad - rightPad}" height="16" rx="8" fill="#e6ecf4"></rect>
          <rect x="${leftPad}" y="${y}" width="${lineWidth}" height="16" rx="8" fill="${fill}"></rect>
          <text class="bar-value" x="${Math.min(leftPad + lineWidth + 8, width - rightPad + 4)}" y="${y + 13}">${esc(formatChartValue(item.label, item.count))}</text>
        `;
      }).join("");
      host.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Столбчатая диаграмма">${rows}</svg>`;
      caption.textContent = options.caption || `Показано значений: ${items.length}.`;
    }

    function renderPreview(preview) {
      if (!preview || !preview.columns || !preview.columns.length) {
        previewMetaEl.textContent = "Не удалось прочитать строки CSV.";
        previewHeadEl.innerHTML = "";
        previewBodyEl.innerHTML = "";
        return;
      }
      previewMetaEl.textContent = `Столбцы: ${preview.columns.join(", ")}. Показано строк: ${preview.rows.length}.`;
      previewHeadEl.innerHTML = `<tr>${preview.columns.map(col => `<th>${esc(col)}</th>`).join("")}</tr>`;
      previewBodyEl.innerHTML = preview.rows.map(row => `
        <tr>${preview.columns.map(col => `<td>${esc(row[col] ?? "")}</td>`).join("")}</tr>
      `).join("");
    }

    // Парсит секцию [3/4] "Приоритеты признаков" из текста отчёта.
    // Строка: "       1  column_unique_ratio      0.4480    10.0%  ████··"
    function parseFeatureImportance(reportText) {
      if (!reportText) return [];
      const re = /^\s+(\d+)\s+([A-Za-z_][A-Za-z0-9_]*)\s+([\d.]+)\s+([\d.]+)%\s+[█·]+\s*$/gm;
      const items = [];
      let m;
      while ((m = re.exec(reportText)) !== null) {
        items.push({
          rank: parseInt(m[1], 10),
          name: m[2],
          std: parseFloat(m[3]),
          pct: parseFloat(m[4]),
        });
      }
      return items;
    }

    // Описания 12 признаков: что они означают для предиката WHERE x op v.
    const FEATURE_DESCRIPTIONS = {
      operator_eq:         "Флаг 0/1: оператор предиката — точное равенство (=).",
      operator_lt:         "Флаг 0/1: оператор предиката — меньше (<, <=).",
      operator_gt:         "Флаг 0/1: оператор предиката — больше (>, >=).",
      operator_between:   "Флаг 0/1: оператор предиката — диапазон (BETWEEN).",
      value_normalized:    "Где сравниваемое значение лежит в диапазоне столбца, нормализовано к [0,1].",
      value2_normalized:   "Для BETWEEN: верхняя граница, нормализованная к [0,1] (для остальных операторов = value_normalized).",
      column_mean:         "Среднее значение столбца, по которому идёт фильтрация.",
      column_std:          "Стандартное отклонение значений столбца — мера разброса данных.",
      column_unique_ratio: "Доля уникальных значений в столбце: 1 — все разные, ~0 — много повторов (категориальный).",
      column_min:          "Минимальное значение в столбце.",
      column_max:          "Максимальное значение в столбце.",
      value_vs_mean:       "Z-score сравниваемого значения: на сколько стандартных отклонений оно отстоит от среднего.",
    };

    function renderFeatureImportance(reportText) {
      const container = document.getElementById("featureImportance");
      const items = parseFeatureImportance(reportText);
      if (!items.length) {
        container.innerHTML = '<p class="fi-empty">Запустите анализ, чтобы увидеть приоритеты признаков.</p>';
        return;
      }
      const maxPct = Math.max.apply(null, items.map(i => i.pct).concat([0.0001]));
      const rows = items.map(it => {
        const widthPct = Math.max(2, (it.pct / maxPct) * 100);
        const cls = it.rank <= 3 ? "fi-top" : "";
        const desc = FEATURE_DESCRIPTIONS[it.name] || "";
        return (
          '<tr class="' + cls + '">' +
          '<td>' + it.rank + '</td>' +
          '<td><code>' + it.name + '</code>' +
            (desc ? '<div class="fi-desc">' + desc + '</div>' : '') +
          '</td>' +
          '<td>' + it.std.toFixed(4) + '</td>' +
          '<td>' + it.pct.toFixed(1) + '%</td>' +
          '<td class="fi-bar-cell"><div class="fi-bar-track">' +
            '<div class="fi-bar-fill" style="width:' + widthPct.toFixed(1) + '%"></div>' +
          '</div></td>' +
          '</tr>'
        );
      }).join("");
      container.innerHTML =
        '<table class="fi-table">' +
        '<thead><tr>' +
          '<th>#</th>' +
          '<th>Признак <span class="th-hint">(имя в модели и описание)</span></th>' +
          '<th title="Стандартное отклонение нормализованного признака в обучающей выборке">Std <span class="th-hint">(?)</span></th>' +
          '<th>Важность <span class="th-hint">(% от суммы Std)</span></th>' +
          '<th></th>' +
        '</tr></thead>' +
        '<tbody>' + rows + '</tbody></table>';
    }

    function safe(label, fn) {
      try { fn(); }
      catch (err) { console.error("[applyState] " + label + " failed:", err); }
    }

    function applyState(data) {
      // Обёртываем каждый рендер в try/catch — иначе ошибка в одном
      // блоке (например, в graph charts) останавливает обновление status,
      // кнопок и других панелей.
      safe("status",    () => { statusEl.textContent = data.status; });
      safe("output",    () => { outputEl.textContent = data.output || "Ожидание запуска..."; });
      safe("report",    () => { reportEl.textContent = data.report || "Отчет пока не найден."; });
      // Все три кнопки управляются единообразно: пока идёт анализ —
      // блокируем, иначе разблокируем (включая compareBtn, который
      // compareBackends() вручную выключает в начале).
      safe("buttons", () => {
        runBtn.disabled     = !!data.running;
        compareBtn.disabled = !!data.running;
        generateBtn.disabled = !!data.running;
      });
      safe("stats",     () => { renderStats(data.summary); });
      safe("features",  () => { renderFeatureImportance(data.report || ""); });
      safe("timing",    () => { renderBarChart("timingChart", "timingCaption", data.charts?.timing_bars, {
        caption: "Время полного цикла анализа (мс). Меньше — лучше."
      }); });
      safe("error",     () => { renderBarChart("errorChart", "errorCaption", data.charts?.error_bars, {
        caption: "Сравнение средних ошибок моделей селективности."
      }); });
      safe("selectivity", () => { renderBarChart("selectivityChart", "selectivityCaption", data.charts?.selectivity_bars, {
        caption: "Средние оценки селективности по всем сгенерированным предикатам."
      }); });
      safe("scan",      () => { renderBarChart("scanChart", "scanCaption", data.charts?.scan_counts, {
        caption: "Сколько раз backend рекомендовал Index Scan, Seq Scan и Bitmap Scan."
      }); });
      safe("preview",   () => { renderPreview(data.preview); });
    }

    // Кнопки "Копировать" / "Скачать" в панелях
    async function copyToClipboard(text, btn) {
      try {
        await navigator.clipboard.writeText(text);
      } catch (err) {
        // fallback для старых браузеров
        const ta = document.createElement("textarea");
        ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
        document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        ta.remove();
      }
      const original = btn.textContent;
      btn.textContent = "Скопировано";
      btn.classList.add("copied");
      setTimeout(() => { btn.textContent = original; btn.classList.remove("copied"); }, 1500);
    }

    function downloadAsText(text, filename) {
      const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = filename;
      document.body.appendChild(a); a.click();
      setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);
    }

    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      // не позволяем кнопкам внутри <summary> сворачивать панель
      if (target.classList.contains("tool-btn")) {
        event.preventDefault();
        event.stopPropagation();
        const copyId = target.getAttribute("data-copy");
        if (copyId) {
          const el = document.getElementById(copyId);
          if (el) copyToClipboard(el.textContent || "", target);
          return;
        }
        const dlId = target.getAttribute("data-download");
        if (dlId) {
          const el = document.getElementById(dlId);
          const filename = target.getAttribute("data-filename") || (dlId + ".txt");
          if (el) downloadAsText(el.textContent || "", filename);
        }
      }
    });

    async function fetchState() {
      if (appMode === "local") {
        applyState(initialState);
        return;
      }
      const response = await fetch("/state", { cache: "no-store" });
      const data = await response.json();
      applyState(data);
    }

    async function runAnalysis(backendOverride) {
      if (appMode === "local") {
        statusEl.textContent = "В локальном standalone-режиме live-запуск недоступен. Используйте browser mode.";
        return;
      }
      const body = new URLSearchParams();
      body.set("dataset", datasetInput.value.trim());
      body.set("backend", backendOverride || selectedBackend());
      const response = await fetch("/run", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body
      });
      const data = await response.json();
      statusEl.textContent = data.status;
      await fetchState();
    }

    async function compareBackends() {
      if (appMode === "local") return;
      compareBtn.disabled = true;
      runBtn.disabled = true;
      statusEl.textContent = "Сравнение: запускаю Rust...";
      const body = new URLSearchParams();
      body.set("dataset", datasetInput.value.trim());
      const response = await fetch("/compare", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body
      });
      const data = await response.json();
      statusEl.textContent = data.status;
    }

    async function loadPreview() {
      if (appMode === "local") {
        applyState(initialState);
        return;
      }
      const body = new URLSearchParams();
      body.set("dataset", datasetInput.value.trim());
      const response = await fetch("/preview", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body
      });
      const data = await response.json();
      statusEl.textContent = data.status;
      await fetchState();
    }

    async function generateCsv() {
      if (appMode === "local") {
        statusEl.textContent = "В standalone-режиме генерация из интерфейса отключена. Используйте browser mode.";
        return;
      }
      const body = new URLSearchParams();
      body.set("dataset", datasetInput.value.trim());
      const response = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body
      });
      const data = await response.json();
      statusEl.textContent = data.status;
      await loadPreview();
    }

    runBtn.addEventListener("click", () => runAnalysis());
    compareBtn.addEventListener("click", compareBackends);
    generateBtn.addEventListener("click", generateCsv);
    defaultBtn.addEventListener("click", () => {
      datasetInput.value = defaultDataset;
      loadPreview();
    });
    refreshBtn.addEventListener("click", loadPreview);
    // авто-обновление превью при ручном изменении пути (debounce ~600 мс),
    // чтобы пользователь не нажимал "Обновить данные" каждый раз
    let datasetDebounce = null;
    datasetInput.addEventListener("input", () => {
      if (appMode === "local") return;
      clearTimeout(datasetDebounce);
      datasetDebounce = setTimeout(loadPreview, 600);
    });
    filePicker.addEventListener("change", async (event) => {
      const [file] = event.target.files || [];
      await uploadSelectedFile(file);
      event.target.value = "";
    });

    if (appMode === "local") {
      runBtn.disabled = true;
      browseBtn.disabled = true;
      const modeHint = document.getElementById("modeHint");
      modeHint.textContent = "Это локальный standalone-дашборд без сервера. Для интерактивного запуска анализа используйте browser mode.";
      modeHint.hidden = false;
      fetchState();
    } else {
      fetchState().then(loadPreview);
      setInterval(fetchState, 1200);
    }
  </script>
</body>
</html>
"""


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.status = "Готово к запуску анализа."
        self.output = ""
        self.report_mtime: float = 0.0
        self.report = self._read_report()
        self.summary = {}
        self.charts = {}
        self.preview = {"columns": [], "rows": []}
        self.timings: dict[str, float] = {}  # {"rust": ms, "python": ms}

    def _read_report(self) -> str:
        if REPORT_FILE.exists():
            try:
                self.report_mtime = REPORT_FILE.stat().st_mtime
            except OSError:
                self.report_mtime = 0.0
            return REPORT_FILE.read_text(encoding="utf-8")
        self.report_mtime = 0.0
        return "Отчет пока не найден.\n\nЗапустите анализ, чтобы создать новый отчет."

    def _refresh_report_if_changed(self) -> bool:
        """Если файл отчёта на диске новее запомненного — перечитываем его
        и пересчитываем summary/charts на основе нового текста.

        Это позволяет UI автоматически подхватывать новый отчёт даже когда
        бинарь запускают мимо GUI (например, из терминала или сетапа), а
        также страхует от случаев, когда run_analysis по какой-то причине
        не успел обновить сводку — пользователю не нужно жать "Обновить".
        """
        try:
            mtime = REPORT_FILE.stat().st_mtime if REPORT_FILE.exists() else 0.0
        except OSError:
            mtime = 0.0
        if mtime == self.report_mtime:
            return False
        self.report = self._read_report()
        # Пересчитываем сводку/графики из свежего текста отчёта.
        # Не передаём dataset_path в parse_report_metrics — иначе при
        # несовпадении путей (например, относительный vs абсолютный)
        # вернутся пустые словари.
        summary, charts = parse_report_metrics(self.report)
        if summary or charts:
            merged_summary = dict(summary)
            merged_charts = dict(charts)
            # Сохраняем накопленные тайминги Rust/Python
            if self.timings.get("rust"):
                merged_summary["rust_ms"] = round(self.timings["rust"], 3)
            if self.timings.get("python"):
                merged_summary["python_ms"] = round(self.timings["python"], 3)
            timing_bars = []
            if self.timings.get("rust"):
                timing_bars.append({"label": "Rust", "count": round(self.timings["rust"], 3)})
            if self.timings.get("python"):
                timing_bars.append({"label": "Python", "count": round(self.timings["python"], 3)})
            if timing_bars:
                merged_charts["timing_bars"] = timing_bars
            # Подхватываем тайминг и backend из свежего отчёта в timings,
            # чтобы при следующем запуске сравнения были актуальные данные.
            last_backend = summary.get("last_backend")
            last_elapsed = summary.get("last_elapsed_ms")
            if last_backend and last_elapsed:
                self.timings[str(last_backend)] = float(last_elapsed)
                merged_summary[f"{last_backend}_ms"] = round(float(last_elapsed), 3)
                # обновляем timing_bars с учётом нового значения
                bars = []
                if self.timings.get("rust"):
                    bars.append({"label": "Rust", "count": round(self.timings["rust"], 3)})
                if self.timings.get("python"):
                    bars.append({"label": "Python", "count": round(self.timings["python"], 3)})
                if bars:
                    merged_charts["timing_bars"] = bars
            self.summary = merged_summary
            self.charts = merged_charts
        return True

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            self._refresh_report_if_changed()
            return {
                "running": self.running,
                "status": self.status,
                "output": self.output,
                "report": self.report,
                "summary": self.summary,
                "charts": self.charts,
                "preview": self.preview,
            }

    def set_status(self, status: str) -> None:
        with self.lock:
            self.status = status

    def set_running(self, running: bool) -> None:
        with self.lock:
            self.running = running

    def reset_output(self) -> None:
        with self.lock:
            self.output = ""

    def append_output(self, chunk: str) -> None:
        with self.lock:
            self.output += chunk

    def reload_report(self) -> None:
        with self.lock:
            self.report = self._read_report()

    def update_preview(self, summary: dict[str, object], charts: dict[str, object]) -> None:
        with self.lock:
            merged_summary = dict(summary)
            merged_charts = dict(charts)
            if self.timings.get("rust"):
                merged_summary["rust_ms"] = round(self.timings["rust"], 3)
            if self.timings.get("python"):
                merged_summary["python_ms"] = round(self.timings["python"], 3)
            timing_bars = []
            if self.timings.get("rust"):
                timing_bars.append({"label": "Rust", "count": round(self.timings["rust"], 3)})
            if self.timings.get("python"):
                timing_bars.append({"label": "Python", "count": round(self.timings["python"], 3)})
            if timing_bars:
                merged_charts["timing_bars"] = timing_bars
            self.summary = merged_summary
            self.charts = merged_charts

    def record_timing(self, backend: str, elapsed_ms: float) -> None:
        with self.lock:
            self.timings[backend] = elapsed_ms

    def set_preview(self, preview: dict[str, object]) -> None:
        with self.lock:
            self.preview = preview


STATE = AppState()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _build_histogram(
    values: list[float],
    bins: int = 8,
    min_value: float | None = None,
    max_value: float | None = None,
) -> list[dict[str, object]]:
    if not values:
        return []
    low = min(values) if min_value is None else min_value
    high = max(values) if max_value is None else max_value
    if high == low:
        return [{"label": f"{low:.0f}", "count": len(values)}]
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        clamped = min(max(value, low), high)
        index = int((clamped - low) / width)
        if index >= bins:
            index = bins - 1
        counts[index] += 1
    result = []
    for index, count in enumerate(counts):
        start = low + width * index
        end = start + width
        result.append({"label": f"{start:.0f}-{end:.0f}", "count": count})
    return result


def _should_show_output_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    hidden_prefixes = (
        "Compiling ",
        "Finished ",
        "Running ",
        "warning:",
        "-->",
        "=",
        "|",
    )
    if stripped.startswith(hidden_prefixes):
        return False
    if stripped.startswith("note:"):
        return False
    if stripped.startswith("help:"):
        return False
    if stripped.startswith("For more information about this error"):
        return False
    if stripped.startswith("thread '"):
        return False
    if stripped.startswith("stack backtrace:"):
        return False
    return True


def resolve_backend_command(dataset_path: Path, backend: str = "rust") -> list[str]:
    if backend == "python":
        script = ROOT_DIR / "python_backend.py"
        if not script.exists():
            raise FileNotFoundError(f"Не найден Python backend: {script}")
        return [sys.executable, str(script), str(dataset_path)]

    release_binary = ROOT_DIR / "target" / "release" / "bac123"
    if release_binary.exists():
        return [str(release_binary), str(dataset_path)]
    compiled_binary = ROOT_DIR / "target" / "debug" / "bac123"
    if compiled_binary.exists():
        return [str(compiled_binary), str(dataset_path)]

    candidates = [
        shutil.which("cargo"),
        str(Path.home() / ".cargo" / "bin" / "cargo"),
        "/opt/homebrew/bin/cargo",
        "/usr/local/bin/cargo",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return [candidate, "run", "--release", "--", str(dataset_path)]

    raise FileNotFoundError(
        "Не найден ни `cargo`, ни готовый бинарник `target/{release,debug}/bac123`."
    )


def render_html_page(server_url: str, app_mode: str, initial_state: dict[str, object]) -> str:
    return (
        HTML_PAGE
        .replace("__DEFAULT_DATASET__", html.escape(str(DEFAULT_DATASET)))
        .replace("__DEFAULT_DATASET_JSON__", json.dumps(str(DEFAULT_DATASET)))
        .replace("__SERVER_URL__", html.escape(server_url))
        .replace("__APP_MODE__", app_mode)
        .replace("__INITIAL_STATE_JSON__", json.dumps(initial_state, ensure_ascii=False))
    )


def generate_synthetic_csv_file(output_path: Path, num_rows: int = 10_000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "age", "salary", "department_id", "score", "years_experience"])
        for index in range(num_rows):
            age = max(18, int(round(random.gauss(35, 10))))
            salary = int(round(50_000 * ((1.0 - random.random()) ** -1.0)))
            department_id = random.randint(1, 20)
            score = round(random.uniform(0, 100), 2)
            experience = random.randint(0, max(age - 22, 1))
            writer.writerow([index, age, salary, department_id, score, experience])


def save_uploaded_csv(filename: str, content: bytes) -> Path:
    safe_name = Path(filename or "uploaded.csv").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(safe_name).stem) or "uploaded"
    destination = UPLOAD_DIR / f"{int(time.time() * 1000)}_{stem}.csv"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def load_csv_preview(dataset_raw: str, max_rows: int = 8) -> dict[str, object]:
    dataset_value = dataset_raw.strip() or str(DEFAULT_DATASET)
    path = Path(dataset_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = []
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            rows.append({column: row.get(column, "") for column in columns})

    return {"columns": columns, "rows": rows}


def parse_report_metrics(report_text: str, dataset_path: Path | None = None) -> tuple[dict[str, object], dict[str, object]]:
    if not report_text.strip():
        return {}, {}

    # Структурированный отчёт ML Optimizer: табличные строки вида
    #   "<pred> <actual> <knn> <hist> <approx> <Seq|Index|Bitmap> Scan"
    table_row_pattern = re.compile(
        r"^\s{4,}(?P<pred>.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(Seq Scan|Index Scan|Bitmap Scan)\s*$"
    )
    numeric_columns_pattern = re.compile(r"Числовые столбцы(?:\s*\(\d+\))?:\s*(.+)")
    dataset_pattern = re.compile(r"Файл(?:\s+данных)?:\s*(.+)")
    backend_pattern = re.compile(r"^\s*Backend:\s*(\w+)", re.MULTILINE)
    elapsed_pattern = re.compile(r"Время выполнения:\s*([0-9.]+)\s*мс")

    values = []
    errors = []
    scan_counts: dict[str, int] = {}
    numeric_columns = 0
    report_dataset: str | None = None

    for line in report_text.splitlines():
        dataset_match = dataset_pattern.search(line)
        if dataset_match:
            report_dataset = dataset_match.group(1).strip()
            continue
        row_match = table_row_pattern.match(line)
        if row_match:
            actual, knn, hist, approx = (float(x) for x in row_match.group(2, 3, 4, 5))
            scan_label = row_match.group(6).strip()
            values.append((actual, knn, hist, approx))
            errors.append((abs(actual - knn), abs(actual - hist), abs(actual - approx)))
            scan_counts[scan_label] = scan_counts.get(scan_label, 0) + 1
            continue
        numeric_match = numeric_columns_pattern.search(line)
        if numeric_match:
            numeric_columns = len([part for part in numeric_match.group(1).split(",") if part.strip()])

    if dataset_path is not None and report_dataset is not None:
        if Path(report_dataset).expanduser().resolve() != dataset_path.resolve():
            return {}, {}

    summary: dict[str, object] = {
        "numeric_columns": numeric_columns,
        "predicate_count": len(values),
    }
    charts: dict[str, object] = {}

    backend_match = backend_pattern.search(report_text)
    elapsed_match = elapsed_pattern.search(report_text)
    if backend_match and elapsed_match:
        summary["last_backend"] = backend_match.group(1).strip().lower()
        summary["last_elapsed_ms"] = float(elapsed_match.group(1))

    if errors:
        avg_knn = sum(item[0] for item in errors) / len(errors)
        avg_hist = sum(item[1] for item in errors) / len(errors)
        avg_approx = sum(item[2] for item in errors) / len(errors)
        summary["avg_error_knn"] = round(avg_knn, 4)
        summary["avg_error_hist"] = round(avg_hist, 4)
        summary["avg_error_approx"] = round(avg_approx, 4)
        charts["error_bars"] = [
            {"label": "KNN", "count": round(avg_knn, 4)},
            {"label": "Hist", "count": round(avg_hist, 4)},
            {"label": "Approx", "count": round(avg_approx, 4)},
        ]

    if values:
        avg_actual = sum(item[0] for item in values) / len(values)
        avg_knn_sel = sum(item[1] for item in values) / len(values)
        avg_hist_sel = sum(item[2] for item in values) / len(values)
        avg_approx_sel = sum(item[3] for item in values) / len(values)
        charts["selectivity_bars"] = [
            {"label": "Actual", "count": round(avg_actual, 4)},
            {"label": "KNN", "count": round(avg_knn_sel, 4)},
            {"label": "Hist", "count": round(avg_hist_sel, 4)},
            {"label": "Approx", "count": round(avg_approx_sel, 4)},
        ]

    if scan_counts:
        charts["scan_counts"] = [
            {"label": key, "count": value}
            for key, value in sorted(scan_counts.items(), key=lambda item: item[1], reverse=True)
        ]

    return summary, charts


def load_dataset_preview(dataset_raw: str) -> tuple[dict[str, object], dict[str, object]]:
    dataset_value = dataset_raw.strip() or str(DEFAULT_DATASET)
    path = Path(dataset_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    numeric_columns = 0
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames or []
        sample: dict[str, list[str]] = {header: [] for header in headers}
        for index, row in enumerate(reader):
            if index >= MAX_ROWS_FOR_PREVIEW:
                break
            for header in headers:
                sample[header].append((row.get(header) or "").strip())

    for values in sample.values():
        clean = [value for value in values if value]
        if clean and all(_is_number(value) for value in clean):
            numeric_columns += 1

    report_text = REPORT_FILE.read_text(encoding="utf-8") if REPORT_FILE.exists() else ""
    summary, charts = parse_report_metrics(report_text, path)
    if not summary:
        summary = {
            "numeric_columns": numeric_columns,
            "predicate_count": 0,
            "avg_error_knn": 0.0,
            "avg_error_hist": 0.0,
            "avg_error_approx": 0.0,
        }
    else:
        summary["numeric_columns"] = numeric_columns or summary.get("numeric_columns", 0)
    return summary, charts


def create_local_dashboard(dataset_raw: str) -> Path:
    summary, charts = load_dataset_preview(dataset_raw)
    report = REPORT_FILE.read_text(encoding="utf-8") if REPORT_FILE.exists() else "Отчет пока не найден."
    preview = load_csv_preview(dataset_raw)
    snapshot = {
        "running": False,
        "status": "Локальный дашборд построен по текущему CSV.",
        "output": "Standalone-режим показывает локальный снимок данных и отчета.\nДля интерактивного запуска Rust-анализа используйте browser mode.",
        "report": report,
        "summary": summary,
        "charts": charts,
        "preview": preview,
    }
    page = render_html_page("local-file", "local", snapshot)
    LOCAL_DASHBOARD_FILE.write_text(page, encoding="utf-8")
    return LOCAL_DASHBOARD_FILE


def run_analysis(dataset_raw: str, backend: str = "rust", *, manage_state: bool = True) -> int:
    dataset_value = dataset_raw.strip() or str(DEFAULT_DATASET)
    dataset_path = Path(dataset_value).expanduser()

    if dataset_path.suffix.lower() != ".csv":
        if manage_state:
            STATE.set_running(False)
            STATE.set_status("Ошибка: нужно указать путь к CSV-файлу.")
        return 1

    if manage_state:
        STATE.reset_output()
        STATE.set_running(True)
        STATE.set_status(f"Выполняется анализ ({backend}) файла: {dataset_path}")
    else:
        STATE.append_output(f"\n--- Запуск backend: {backend} ---\n")
        STATE.set_status(f"Сравнение: выполняется backend={backend}")

    try:
        command = resolve_backend_command(dataset_path, backend)
        env = os.environ.copy()
        env["PATH"] = ":".join(
            part for part in [
                env.get("PATH", ""),
                str(Path.home() / ".cargo" / "bin"),
                "/opt/homebrew/bin",
                "/usr/local/bin",
            ] if part
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except FileNotFoundError as exc:
        if manage_state:
            STATE.set_running(False)
        STATE.set_status(f"Ошибка: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover
        if manage_state:
            STATE.set_running(False)
        STATE.set_status(f"Ошибка запуска: {exc}")
        return 1

    assert process.stdout is not None
    for line in process.stdout:
        if _should_show_output_line(line):
            STATE.append_output(line)

    code = process.wait()
    STATE.reload_report()
    if code == 0:
        elapsed_summary, _ = parse_report_metrics(STATE.snapshot()["report"], dataset_path)
        elapsed_ms = elapsed_summary.get("last_elapsed_ms")
        report_backend = elapsed_summary.get("last_backend")
        if elapsed_ms and report_backend:
            STATE.record_timing(str(report_backend), float(elapsed_ms))
        try:
            summary, charts = load_dataset_preview(str(dataset_path))
            STATE.update_preview(summary, charts)
            STATE.set_preview(load_csv_preview(str(dataset_path)))
        except Exception:
            pass
        STATE.set_status(
            f"Анализ ({backend}) завершен. Время: {elapsed_ms:.1f} мс."
            if elapsed_ms else f"Анализ ({backend}) завершен."
        )
    else:
        STATE.set_status(f"Процесс ({backend}) завершился с кодом {code}.")

    if manage_state:
        STATE.set_running(False)
    return code


def run_comparison(dataset_raw: str) -> None:
    STATE.reset_output()
    STATE.set_running(True)
    try:
        for backend in ("rust", "python"):
            run_analysis(dataset_raw, backend=backend, manage_state=False)
        rust_ms = STATE.timings.get("rust")
        python_ms = STATE.timings.get("python")
        if rust_ms and python_ms:
            STATE.set_status("Сравнение завершено.")
        else:
            STATE.set_status("Сравнение завершено, но не удалось получить оба времени.")
    finally:
        STATE.set_running(False)


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            server_url = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
            page = render_html_page(server_url, "browser", STATE.snapshot())
            encoded = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            # отдаём страницу без кэша, иначе браузер держит старый JS/CSS
            # и пользователю приходится делать Cmd+Shift+R после правок UI
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return

        if self.path.startswith("/state"):
            payload = json.dumps(STATE.snapshot(), ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/health":
            payload = b"ok"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path == "/upload":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._write_json(
                    {"ok": False, "status": "Ожидалась загрузка файла в формате multipart/form-data."},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                        "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                    },
                )
                upload_field = form["dataset_file"] if "dataset_file" in form else None
                if upload_field is None or not getattr(upload_field, "file", None):
                    raise ValueError("Файл не был передан.")

                filename = upload_field.filename or "uploaded.csv"
                if not filename.lower().endswith(".csv"):
                    raise ValueError("Нужно выбрать CSV-файл.")

                content = upload_field.file.read()
                if not content:
                    raise ValueError("Выбран пустой файл.")

                saved_path = save_uploaded_csv(filename, content)
                summary, charts = load_dataset_preview(str(saved_path))
                STATE.update_preview(summary, charts)
                STATE.set_preview(load_csv_preview(str(saved_path)))
                self._write_json(
                    {
                        "ok": True,
                        "status": f"Файл загружен через Finder: {saved_path.name}",
                        "dataset": str(saved_path),
                    },
                    HTTPStatus.OK,
                )
            except Exception as exc:
                self._write_json({"ok": False, "status": f"Не удалось загрузить CSV: {exc}"}, HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/preview":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            form = urllib.parse.parse_qs(raw_body)
            dataset = form.get("dataset", [""])[0]
            try:
                summary, charts = load_dataset_preview(dataset)
                STATE.update_preview(summary, charts)
                STATE.set_preview(load_csv_preview(dataset))
                self._write_json({"ok": True, "status": "Графики обновлены."}, HTTPStatus.OK)
            except Exception as exc:
                self._write_json({"ok": False, "status": f"Не удалось построить графики: {exc}"}, HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/generate":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            form = urllib.parse.parse_qs(raw_body)
            dataset = form.get("dataset", [""])[0].strip() or str(DEFAULT_DATASET)
            path = Path(dataset).expanduser()
            try:
                generate_synthetic_csv_file(path)
                STATE.set_preview(load_csv_preview(str(path)))
                summary, charts = load_dataset_preview(str(path))
                STATE.update_preview(summary, charts)
                self._write_json({"ok": True, "status": f"CSV сгенерирован: {path}"}, HTTPStatus.OK)
            except Exception as exc:
                self._write_json({"ok": False, "status": f"Не удалось сгенерировать CSV: {exc}"}, HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/compare":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            form = urllib.parse.parse_qs(raw_body)
            dataset = form.get("dataset", [""])[0]
            if STATE.snapshot()["running"]:
                self._write_json({"ok": False, "status": "Анализ уже выполняется."}, HTTPStatus.CONFLICT)
                return
            thread = threading.Thread(target=run_comparison, args=(dataset,), daemon=True)
            thread.start()
            self._write_json({"ok": True, "status": "Сравнение Rust vs Python запущено."}, HTTPStatus.OK)
            return

        if self.path != "/run":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")
        form = urllib.parse.parse_qs(raw_body)
        dataset = form.get("dataset", [""])[0]
        backend = form.get("backend", ["rust"])[0].lower()
        if backend not in ("rust", "python"):
            backend = "rust"

        if STATE.snapshot()["running"]:
            self._write_json({"ok": False, "status": "Анализ уже выполняется."}, HTTPStatus.CONFLICT)
            return

        thread = threading.Thread(target=run_analysis, args=(dataset, backend), daemon=True)
        thread.start()
        self._write_json({"ok": True, "status": f"Запуск анализа ({backend}) начат."}, HTTPStatus.OK)

    def _write_json(self, payload: dict[str, object], status: HTTPStatus) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


def find_free_port(host: str, start_port: int) -> int:
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError("Не удалось найти свободный порт для web-интерфейса.")


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Optimizer GUI")
    parser.add_argument("--mode", choices=("browser", "local"), default="browser")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=START_PORT)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    if args.mode == "local":
        dashboard_path = create_local_dashboard(args.dataset)
        print("Локальный дашборд создан.")
        print(f"Откройте файл: {dashboard_path}")
        webbrowser.open(dashboard_path.as_uri())
        return

    port = args.port if args.port > 0 else find_free_port(args.host, START_PORT)
    server = ThreadingHTTPServer((args.host, port), RequestHandler)
    url = f"http://{args.host}:{port}"

    print("ML Optimizer GUI запущен.")
    print(f"Откройте в браузере: {url}")

    if not args.no_open:
        threading.Thread(target=lambda: (time.sleep(0.4), webbrowser.open(url)), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка сервера...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
