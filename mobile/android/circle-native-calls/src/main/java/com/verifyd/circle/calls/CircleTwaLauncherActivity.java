package com.verifyd.circle.calls;

import android.content.ComponentName;
import android.net.Uri;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.browser.customtabs.CustomTabsCallback;
import androidx.browser.customtabs.CustomTabsClient;
import androidx.browser.customtabs.CustomTabsService;
import androidx.browser.customtabs.CustomTabsServiceConnection;
import androidx.browser.customtabs.CustomTabsSession;
import androidx.browser.trusted.TrustedWebActivityIntentBuilder;

/** Replacement source for the Circle *launcher only*, using the existing /voice URL. */
public final class CircleTwaLauncherActivity extends AppCompatActivity {
    private static final Uri URL = Uri.parse("https://vfvid.com/voice");
    private CustomTabsClient client;
    private CustomTabsSession session;
    private CustomTabsServiceConnection connection;
    private final CircleTwaBridge bridge = new CircleTwaBridge(this);

    private final CustomTabsCallback callbacks = new CustomTabsCallback() {
        @Override public void onRelationshipValidationResult(
                int relation, Uri requestedOrigin, boolean result, Bundle extras) {
            bridge.onRelationshipValidationResult(relation, requestedOrigin, result);
        }
        @Override public void onNavigationEvent(int event, Bundle extras) {
            if (event == NAVIGATION_FINISHED) bridge.onNavigationFinished();
        }
        @Override public void onMessageChannelReady(Bundle extras) {
            bridge.onMessageChannelReady();
        }
        @Override public void onPostMessage(String message, Bundle extras) {
            bridge.onPostMessage(message);
        }
    };

    @Override protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        String browser = CustomTabsClient.getPackageName(this, null);
        if (browser == null) {
            // A TWA-capable browser is required; never silently switch to an
            // insecure token-bearing WebView or bypass DAL.
            finish();
            return;
        }
        connection = new CustomTabsServiceConnection() {
            @Override public void onCustomTabsServiceConnected(ComponentName name, CustomTabsClient c) {
                client = c;
                client.warmup(0);
                session = client.newSession(callbacks);
                if (session == null) { finish(); return; }
                bridge.attach(session);
                session.validateRelationship(CustomTabsService.RELATION_USE_AS_ORIGIN,
                        CircleTwaBridge.ORIGIN, null);
                new TrustedWebActivityIntentBuilder(URL).build(session)
                        .launchTrustedWebActivity(CircleTwaLauncherActivity.this);
                // Keep this activity alive so the validated Custom Tabs session
                // and PostMessage callback survive while the TWA is foregrounded.
            }
            @Override public void onServiceDisconnected(ComponentName name) {
                client = null;
                session = null;
                bridge.attach(null);
            }
        };
        CustomTabsClient.bindCustomTabsService(this, browser, connection);
    }

    @Override protected void onDestroy() {
        if (connection != null) try { unbindService(connection); }
        catch (IllegalArgumentException ignored) { }
        super.onDestroy();
    }
}
