package com.verifyd.circle.calls;

import android.content.Context;
import android.content.SharedPreferences;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Base64;
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

final class CircleCallApi {
    private static final String KEY = "verifyd-circle-native-calls";
    private final SharedPreferences prefs;
    private final String origin;

    CircleCallApi(Context context) {
        prefs = context.getSharedPreferences(KEY, Context.MODE_PRIVATE);
        origin = "https://verifyd-backend.onrender.com";
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
        if (!configured() || !CircleNativeCallPayload.validId(callId)) {
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

    /** Metadata only: no account/session bearer or FCM token leaves the Android host. */
    JSONObject bridgeHello() throws Exception {
        java.security.KeyPair pair = enrollmentKey();
        JSONObject json = new JSONObject();
        json.put("schema", "verifyd.twa.native-bridge.v1");
        json.put("type", "hello");
        json.put("installation_id", installationId());
        json.put("package_name", "com.verifyd.circle");
        json.put("public_key_spki_b64", b64url(pair.getPublic().getEncoded()));
        return json;
    }

    private static String b64url(byte[] value) {
        return Base64.encodeToString(value,
                Base64.URL_SAFE | Base64.NO_WRAP | Base64.NO_PADDING);
    }

    private java.security.KeyPair enrollmentKey() throws Exception {
        final String alias = KEY + "-ec-p256";
        KeyStore store = KeyStore.getInstance("AndroidKeyStore");
        store.load(null);
        if (store.containsAlias(alias)) {
            return new java.security.KeyPair(store.getCertificate(alias).getPublicKey(),
                    (java.security.PrivateKey) store.getKey(alias, null));
        }
        java.security.KeyPairGenerator generator =
                java.security.KeyPairGenerator.getInstance("EC", "AndroidKeyStore");
        generator.initialize(new KeyGenParameterSpec.Builder(alias,
                KeyProperties.PURPOSE_SIGN | KeyProperties.PURPOSE_VERIFY)
                .setAlgorithmParameterSpec(new java.security.spec.ECGenParameterSpec("secp256r1"))
                .setDigests(KeyProperties.DIGEST_SHA256).build());
        return generator.generateKeyPair();
    }

    /** Redeem web-authorized, single-use 60-second signed challenge. NO web bearer. */
    void redeemIntent(String id, String challenge, String fcmToken) throws Exception {
        if (!"com.verifyd.circle".equals(prefs.getString("circle_package", "com.verifyd.circle")))
            throw new IllegalStateException("Invalid installed Circle app");
        if (!id.matches("nbi_[A-Za-z0-9_-]{32}") || !challenge.matches("[A-Za-z0-9_-]{43}"))
            throw new IllegalArgumentException("Invalid enrollment intent");
        java.security.KeyPair pair = enrollmentKey();
        byte[] encoded = pair.getPublic().getEncoded();
        java.security.MessageDigest sha = java.security.MessageDigest.getInstance("SHA-256");
        byte[] digest = sha.digest(encoded);
        StringBuilder hash = new StringBuilder(64);
        for (byte b : digest) hash.append(String.format(java.util.Locale.ROOT, "%02x", b & 0xff));
        String message = "verifyd.native-bridge.android.v1\\n" + id + "\\n"
                + challenge + "\\n" + installationId() + "\\n"
                + "com.verifyd.circle" + "\\n" + hash;
        java.security.Signature signer = java.security.Signature.getInstance("SHA256withECDSA");
        signer.initSign(pair.getPrivate());
        signer.update(message.getBytes(StandardCharsets.US_ASCII));
        JSONObject body = new JSONObject()
                .put("installation_id", installationId())
                .put("package_name", "com.verifyd.circle")
                .put("public_key_spki_b64", b64url(encoded))
                .put("intent_id", id)
                .put("challenge", challenge)
                .put("signature_b64", b64url(signer.sign()))
                .put("fcm_token", fcmToken)
                .put("opted_in", true);
        JSONObject result = request("POST", "/native-bridge/android/intents/redeem", "", body);
        if (!"verifyd.native-bridge.android.v1".equals(result.optString("schema")))
            throw new IllegalStateException("Enrollment schema mismatch");
        saveDeviceToken(result.getString("device_token"));
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
        if (!configured()) throw new IllegalStateException("Circle API is unconfigured");
        HttpURLConnection connection = (HttpURLConnection)
                URI.create(origin + "/trust-voice" + path).toURL().openConnection();
        try {
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(3000);
            connection.setReadTimeout(3000);
            connection.setRequestMethod(method);
            if (bearer != null && !bearer.isEmpty()) connection.setRequestProperty("Authorization", "Bearer " + bearer);
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
