/* VeriFYD Access — on-demand, current-tab only. No all-site permissions. */
async function toggleOnTab(tab) {
  if (!tab || !Number.isInteger(tab.id)) return;
  if (!/^https?:\/\//i.test(tab.url || '')) {
    console.info('VeriFYD Access: open an ordinary http(s) website first.');
    return;
  }
  try {
    await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: ['content.css'] });
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ['navigation-core.js', 'content.js'],
      world: 'ISOLATED',
    });
  } catch (err) {
    console.warn('VeriFYD Access could not be started on this page:', err?.message || 'unknown error');
  }
}

chrome.action.onClicked.addListener(toggleOnTab);
chrome.commands.onCommand.addListener(async (command) => {
  if (command !== 'toggle-verifiyd-access') return;
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  await toggleOnTab(tabs[0]);
});
