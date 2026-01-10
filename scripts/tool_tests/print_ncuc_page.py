from playwright.sync_api import sync_playwright

URL = "https://starw1.ncuc.gov/NCUC/page/Dockets/portal.aspx"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=30000)

        html = page.content()
        print(html)

        with open("/tmp/ncuc_page.html", "w", encoding="utf-8") as handle:
            handle.write(html)

        browser.close()


if __name__ == "__main__":
    main()
