package com.verifyd.app.calls;

import android.content.Context;
import android.content.SharedPreferences;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Base64;
import com.verifyd.app.R;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.KeyStore;
import java.util.UUID;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;
import org.json.JSONObject;

final class PilotCallApi {
    private static final String KEY = "verifyd-detector-call-pilot";
    private final SharedPreferences prefs;
    private final String origin;

    PilotCallApi(Context context) {
        prefs = context.getSharedPreferences(KEY, Context.MODE_PRIVATE);
        origin = context.getString(R.string.detector_pilot_api_origin).trim();
    }

    boolean configured() {
        try {
            URI uri = URI.create(origin);
            return "https".equals(uri.getScheme()) && uri.getHost() != null
                    && uri.getUserInfo() == null && uri.getRawPath().isEmpty()
                    && uri.getRawQuery() == null && uri.getRawFragment() == null;
        } catch (IllegalArgumentException ignored) {
            return false;
        }
    }

    String mediaUrl(String callId) {
        if (!configured() || !NativeCallPayload.validId(callId)) {
            throw new IllegalArgumentException("Invalid media endpoint");
        }
        return "wss" + origin.substring(5) + "/trust-voice/android-calls/" + callId + "/media";
    }

    boolean enrolled() {
        return configured() && origin.equals(prefs.getString("device_origin", ""))
                && prefs.contains("device_token");
    }

    private String installationId() {
        String id = prefs.getString("installation_id", null);
        if (id != null) return id;
        id = UUID.randomUUID().toString();
        if (!prefs.edit().putString("installation_id", id).commit()) {
            throw new IllegalStateException("Could not save installation ID");
        }
        return id;
    }

    void pair(String code, String fcmToken, String packageName) throws Exception {
        JSONObject body = new JSONObject();
        body.put("installation_id", installationId());
        body.put("package_name", packageName);
        body.put("fcm_token", fcmToken);
        body.put("opted_in", true);
        String token = request("POST", "/android-call-pairings/redeem", code, body)
                .getString("device_token");
        saveDeviceToken(token);
    }

    void unlink() throws Exception {
        try {
            request("DELETE", "/android-call-devices/current", deviceToken(), null);
        } catch (ApiError error) {
            if (error.status != 401) throw error;
        }
        prefs.edit().remove("device_token").apply();
    }

    void refresh(String fcmToken) throws Exception {
        JSONObject body = new JSONObject();
        body.put("fcm_token", fcmToken);
        request("PUT", "/android-call-devices/current/token", deviceToken(), body);
    }

    void decline(String callId) throws Exception {
        request("POST", "/android-calls/" + callId + "/decline", deviceToken(), null);
    }

    private SecretKey storageKey(boolean create) throws Exception {
        KeyStore store = KeyStore.getInstance("AndroidKeyStore");
        store.load(null);
        SecretKey key = (SecretKey) store.getKey(KEY, null);
        if (key != null || !create) return key;
        KeyGenerator generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
        generator.init(new KeyGenParameterSpec.Builder(KEY,
                KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).build());
        return generator.generateKey();
    }

    private void saveDeviceToken(String token) throws Exception {
        if (!token.matches("ac_[A-Za-z0-9_-]{43}")) throw new IllegalStateException("Invalid credential");
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(Cipher.ENCRYPT_MODE, storageKey(true));
        String stored = Base64.encodeToString(cipher.getIV(), Base64.NO_WRAP) + "."
                + Base64.encodeToString(cipher.doFinal(token.getBytes(StandardCharsets.UTF_8)), Base64.NO_WRAP);
        if (!prefs.edit().putString("device_token", stored).putString("device_origin", origin).commit()) {
            throw new IllegalStateException("Could not save pairing");
        }
    }

    String deviceToken() throws Exception {
        if (!enrolled()) throw new IllegalStateException("Device is not paired to this origin");
        String[] stored = prefs.getString("device_token", "").split("\\.", 2);
        if (stored.length != 2) throw new IllegalStateException("Device is not paired");
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(Cipher.DECRYPT_MODE, storageKey(false),
                new GCMParameterSpec(128, Base64.decode(stored[0], Base64.NO_WRAP)));
        return new String(cipher.doFinal(Base64.decode(stored[1], Base64.NO_WRAP)), StandardCharsets.UTF_8);
    }

    private JSONObject request(String method, String path, String bearer, JSONObject body) throws Exception {
        if (!configured()) throw new IllegalStateException("Pilot API is unconfigured");
        HttpURLConnection connection = (HttpURLConnection)
                URI.create(origin + "/trust-voice" + path).toURL().openConnection();
        try {
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(3000);
            connection.setReadTimeout(3000);
            connection.setRequestMethod(method);
            connection.setRequestProperty("Authorization", "Bearer " + bearer);
            if (body != null) {
                connection.setDoOutput(true);
                connection.setRequestProperty("Content-Type", "application/json");
                try (java.io.OutputStream output = connection.getOutputStream()) {
                    output.write(body.toString().getBytes(StandardCharsets.UTF_8));
                }
            }
            int status = connection.getResponseCode();
            if (status < 200 || status >= 300) throw new ApiError(status);
            try (InputStream input = connection.getInputStream();
                 ByteArrayOutputStream output = new ByteArrayOutputStream()) {
                byte[] buffer = new byte[512];
                int count;
                while ((count = input.read(buffer)) != -1) {
                    if (output.size() + count > 4096) throw new IllegalStateException("Unexpected response size");
                    output.write(buffer, 0, count);
                }
                return new JSONObject(new String(output.toByteArray(), StandardCharsets.UTF_8));
            }
        } finally {
            connection.disconnect();
        }
    }

    static final class ApiError extends Exception {
        final int status;
        ApiError(int status) { this.status = status; }
    }
}
