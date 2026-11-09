package com.verifyd.app.calls;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.text.InputType;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import com.google.android.gms.tasks.Tasks;
import com.google.firebase.messaging.FirebaseMessaging;
import java.util.concurrent.TimeUnit;

public final class PilotPairActivity extends Activity {
    private EditText codeInput;
    private TextView status;
    private Button pairButton;
    private Button unlinkButton;
    private PilotCallApi api;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_SECURE);
        api = new PilotCallApi(this);
        LinearLayout screen = new LinearLayout(this);
        screen.setOrientation(LinearLayout.VERTICAL);
        int padding = (int) (24 * getResources().getDisplayMetrics().density);
        screen.setPadding(padding, padding, padding, padding);
        status = new TextView(this);
        screen.addView(status);
        codeInput = new EditText(this);
        codeInput.setHint("Paste VeriFYD pairing code");
        codeInput.setSingleLine();
        codeInput.setSaveEnabled(false);
        codeInput.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        screen.addView(codeInput);
        pairButton = new Button(this);
        pairButton.setText("Pair this phone");
        pairButton.setOnClickListener(view -> pair());
        screen.addView(pairButton);
        unlinkButton = new Button(this);
        unlinkButton.setText("Unlink this phone");
        unlinkButton.setOnClickListener(view -> run(() -> api.unlink(), "Phone unlinked."));
        screen.addView(unlinkButton);
        setContentView(screen);
        refreshButtons();
        status.setText(!api.configured() ? "Set the isolated pilot API origin before building."
                : api.enrolled() ? "This phone is enrolled." : "Generate a code from the pilot backend.");
        if (api.enrolled()) {
            run(() -> {
                String token = Tasks.await(FirebaseMessaging.getInstance().getToken(), 10, TimeUnit.SECONDS);
                api.refresh(token);
            }, "This phone is enrolled.");
        }
    }

    private void pair() {
        if (Build.VERSION.SDK_INT < 26) {
            status.setText("The incoming-call pilot requires Android 8 or newer.");
            return;
        }
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[] {Manifest.permission.POST_NOTIFICATIONS}, 1);
            status.setText("Allow notifications, then tap Pair again.");
            return;
        }
        String code = codeInput.getText().toString().trim();
        if (!code.matches("ap_[A-Za-z0-9_-]{43}")) {
            status.setText("Paste a valid VeriFYD pairing code.");
            return;
        }
        codeInput.setText("");
        run(() -> {
            String token = Tasks.await(FirebaseMessaging.getInstance().getToken(), 10, TimeUnit.SECONDS);
            api.pair(code, token, getPackageName());
        }, "Phone paired for incoming-call pilot notifications.");
    }

    private void run(PilotOperation operation, String success) {
        pairButton.setEnabled(false);
        unlinkButton.setEnabled(false);
        status.setText("Working…");
        new Thread(() -> {
            String message;
            try {
                operation.run();
                message = success;
            } catch (PilotCallApi.ApiError error) {
                message = error.status == 401 ? "Pairing expired or was revoked. Generate a fresh code."
                        : "Pilot server rejected the request (" + error.status + ").";
            } catch (Exception ignored) {
                message = "Check the pilot connection and Firebase configuration, then retry.";
            }
            String result = message;
            runOnUiThread(() -> {
                if (isDestroyed()) return;
                status.setText(result);
                refreshButtons();
            });
        }).start();
    }

    private void refreshButtons() {
        pairButton.setEnabled(api.configured());
        unlinkButton.setEnabled(api.enrolled());
    }

    private interface PilotOperation { void run() throws Exception; }
}
