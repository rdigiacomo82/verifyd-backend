package com.verifyd.circle.calls;

import android.Manifest;
import android.app.Activity;
import android.app.NotificationManager;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.net.Uri;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

/** User-visible opt-in permission screen; never auto-unlocks the phone. */
public final class CircleNotificationsPermissionActivity extends Activity {
    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        int pad = Math.round(24 * getResources().getDisplayMetrics().density);
        LinearLayout column = new LinearLayout(this);
        column.setPadding(pad, pad, pad, pad);
        column.setOrientation(LinearLayout.VERTICAL);
        TextView title = new TextView(this);
        title.setText("Enable VeriFYD Circle incoming calls");
        title.setTextSize(20);
        column.addView(title);
        Button allow = new Button(this);
        allow.setText("Allow notifications");
        allow.setMinHeight(pad * 2);
        allow.setOnClickListener(v -> {
            if (Build.VERSION.SDK_INT >= 33)
                requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 11);
            else finish();
        });
        column.addView(allow);
        if (Build.VERSION.SDK_INT >= 34) {
            NotificationManager notifications = getSystemService(NotificationManager.class);
            if (notifications != null && !notifications.canUseFullScreenIntent()) {
                TextView explanation = new TextView(this);
                explanation.setText("For the incoming-call screen to appear while locked, allow full-screen notifications in Android settings. Otherwise you may see a heads-up alert.");
                column.addView(explanation);
                Button fullScreen = new Button(this);
                fullScreen.setText("Review full-screen call permission");
                fullScreen.setMinHeight(pad * 2);
                fullScreen.setOnClickListener(v -> startActivity(new Intent(
                        Settings.ACTION_MANAGE_APP_USE_FULL_SCREEN_INTENT,
                        Uri.parse("package:" + getPackageName()))));
                column.addView(fullScreen);
            }
        }
        Button close = new Button(this);
        close.setText("Continue to VeriFYD");
        close.setMinHeight(pad * 2);
        close.setOnClickListener(v -> finish());
        column.addView(close);
        setContentView(column);
    }
}
