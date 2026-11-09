package com.verifyd.app.calls;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

/** User-visible entry point for while-in-use microphone permission. */
public final class PilotAnswerActivity extends Activity {
    private boolean attempted;
    private TextView status;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        attempted = state != null && state.getBoolean("attempted");
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        status = new TextView(this);
        status.setText("VeriFYD audio call");
        layout.addView(status);
        Button close = new Button(this);
        close.setText("Close");
        close.setOnClickListener(view -> finish());
        layout.addView(close);
        setContentView(layout);
    }

    @Override protected void onSaveInstanceState(Bundle state) {
        state.putBoolean("attempted", attempted);
        super.onSaveInstanceState(state);
    }

    @Override public void onWindowFocusChanged(boolean focused) {
        super.onWindowFocusChanged(focused);
        if (!focused || attempted) return;
        attempted = true;
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            startAudio();
        } else {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 1);
        }
    }

    @Override public void onRequestPermissionsResult(int request, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(request, permissions, results);
        if (request != 1) return;
        if (results.length == 1 && results[0] == PackageManager.PERMISSION_GRANTED) {
            attempted = false;
            if (hasWindowFocus()) onWindowFocusChanged(true);
        } else {
            status.setText("Microphone access is required to answer.");
        }
    }

    private void startAudio() {
        String callId = getIntent().getStringExtra("call_id");
        long expiry = getIntent().getLongExtra("expires_at", 0);
        long now = System.currentTimeMillis() / 1000;
        if (!NativeCallPayload.validId(callId) || expiry <= now || expiry > now + 30
                || !new PilotCallApi(this).enrolled()) {
            status.setText("This call is no longer available.");
            return;
        }
        try {
            startForegroundService(new Intent(this, PilotAudioService.class)
                    .putExtra("call_id", callId).putExtra("expires_at", expiry));
            status.setText("Connecting. Use the ongoing call notification to hang up.");
        } catch (RuntimeException ignored) {
            status.setText("Android could not start the microphone service.");
        }
    }
}
