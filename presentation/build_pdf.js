// Converte slides.html in presentazione.pdf (Chromium headless via Playwright).
const path = require('path');
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await page.goto('file://' + path.join(__dirname, 'slides.html'), { waitUntil: 'networkidle' });
  await page.pdf({ path: path.join(__dirname, 'presentazione.pdf'), width: '1280px', height: '720px', printBackground: true });
  await browser.close();
})();
