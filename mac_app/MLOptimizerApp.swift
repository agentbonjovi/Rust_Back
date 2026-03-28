import Cocoa
import Foundation
import WebKit

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var backendProcess: Process?
    private let host = "127.0.0.1"
    private let port = 8765

    func applicationDidFinishLaunching(_ notification: Notification) {
        createWindow()
        showLoadingPage(message: "Запуск backend и подготовка интерфейса...")
        startBackend()
        pollServerAndLoad()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        guard let process = backendProcess, process.isRunning else { return }
        process.terminate()
    }

    private func createWindow() {
        let rect = NSRect(x: 0, y: 0, width: 1320, height: 920)
        window = NSWindow(
            contentRect: rect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.center()
        window.title = "ML Optimizer"
        window.makeKeyAndOrderFront(nil)

        let configuration = WKWebViewConfiguration()
        webView = WKWebView(frame: rect, configuration: configuration)
        webView.navigationDelegate = self
        webView.autoresizingMask = [.width, .height]
        window.contentView = webView
    }

    private func projectRoot() -> URL {
        URL(fileURLWithPath: Bundle.main.bundlePath).deletingLastPathComponent()
    }

    private func backendScriptURL() -> URL {
        projectRoot().appendingPathComponent("gui_app.py")
    }

    private func startBackend() {
        let scriptURL = backendScriptURL()
        guard FileManager.default.fileExists(atPath: scriptURL.path) else {
            showErrorPage(message: "Не найден gui_app.py рядом с приложением.")
            return
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.currentDirectoryURL = projectRoot()
        process.arguments = [
            scriptURL.path,
            "--mode", "browser",
            "--host", host,
            "--port", String(port),
            "--no-open",
        ]

        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe

        do {
            try process.run()
            backendProcess = process
        } catch {
            backendProcess = nil
        }
    }

    private func pollServerAndLoad(attempt: Int = 0) {
        let healthURL = URL(string: "http://\(host):\(port)/health")!
        var request = URLRequest(url: healthURL)
        request.timeoutInterval = 1.0

        URLSession.shared.dataTask(with: request) { [weak self] data, response, _ in
            guard let self else { return }
            let http = response as? HTTPURLResponse
            let isHealthy = http?.statusCode == 200 && String(data: data ?? Data(), encoding: .utf8) == "ok"

            DispatchQueue.main.async {
                if isHealthy {
                    let url = URL(string: "http://\(self.host):\(self.port)")!
                    self.webView.load(URLRequest(url: url))
                    return
                }

                if attempt >= 40 {
                    self.showErrorPage(message: "Не удалось поднять локальный GUI backend. Проверьте, что рядом с приложением есть gui_app.py и установлен Python 3.")
                    return
                }

                self.showLoadingPage(message: "Ожидание запуска локального сервера...")
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    self.pollServerAndLoad(attempt: attempt + 1)
                }
            }
        }.resume()
    }

    private func showLoadingPage(message: String) {
        webView.loadHTMLString(htmlPage(title: "Запуск приложения", body: message, accent: "#0f766e"), baseURL: nil)
    }

    private func showErrorPage(message: String) {
        webView.loadHTMLString(htmlPage(title: "Ошибка запуска", body: message, accent: "#b91c1c"), baseURL: nil)
    }

    private func htmlPage(title: String, body: String, accent: String) -> String {
        """
        <!doctype html>
        <html lang="ru">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            body {
              margin: 0;
              min-height: 100vh;
              display: grid;
              place-items: center;
              font-family: -apple-system, BlinkMacSystemFont, sans-serif;
              background: linear-gradient(160deg, #f5efe5, #fbf8f3);
              color: #1f2937;
            }
            .card {
              width: min(620px, calc(100vw - 48px));
              padding: 28px;
              border-radius: 22px;
              background: rgba(255,255,255,0.88);
              box-shadow: 0 18px 60px rgba(31, 41, 55, 0.10);
              border: 1px solid rgba(0,0,0,0.06);
            }
            h1 { margin: 0 0 10px; font-size: 30px; }
            p { margin: 0; line-height: 1.55; color: #4b5563; }
            .bar {
              width: 84px;
              height: 8px;
              border-radius: 999px;
              background: \(accent);
              margin-bottom: 16px;
            }
          </style>
        </head>
        <body>
          <div class="card">
            <div class="bar"></div>
            <h1>\(title)</h1>
            <p>\(body)</p>
          </div>
        </body>
        </html>
        """
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.setActivationPolicy(.regular)
app.delegate = delegate
app.run()
