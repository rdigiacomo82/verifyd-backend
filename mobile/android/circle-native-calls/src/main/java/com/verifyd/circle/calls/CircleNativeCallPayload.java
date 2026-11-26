package com.verifyd.circle.calls;

import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.UUID;
import java.util.regex.Pattern;

/** Server's original 30-second deadline is authoritative. Never extend on receipt. */
final class CircleNativeCallPayload {
    private static final UUID NAMESPACE = UUID.fromString("fba684f5-f324-4e32-9329-63b0f6632691");
    private static final Pattern ID = Pattern.compile("^tvcall_[0-9a-f]{32}$");
    final String id;
    final String type;
    final long expirySeconds;

    private CircleNativeCallPayload(String id, String type, long expirySeconds) {
        this.id = id; this.type = type; this.expirySeconds = expirySeconds;
    }

    static boolean validId(String id) { return id != null && ID.matcher(id).matches(); }

    static CircleNativeCallPayload parse(Map<String,String> data, long now) {
        if (!"verifyd.native_call.v1".equals(data.get("schema"))
                || !"incoming_call".equals(data.get("kind"))) return null;
        String id = data.get("call_id");
        String type = data.get("call_type");
        if (!validId(id) || !("audio".equals(type) || "video".equals(type))) return null;
        try {
            long deadline = Long.parseLong(data.get("expires_at"));
            if (deadline <= now || deadline > now + 30) return null;
            String expected = uuid5(NAMESPACE, id);
            if (!expected.equals(data.get("call_uuid"))) return null;
            return new CircleNativeCallPayload(id, type, deadline);
        } catch (RuntimeException ignored) { return null; }
    }

    private static String uuid5(UUID namespace, String name) {
        try {
            java.nio.ByteBuffer bytes = java.nio.ByteBuffer.allocate(16);
            bytes.putLong(namespace.getMostSignificantBits());
            bytes.putLong(namespace.getLeastSignificantBits());
            java.security.MessageDigest sha1 = java.security.MessageDigest.getInstance("SHA-1");
            sha1.update(bytes.array());
            byte[] digest = sha1.digest(name.getBytes(StandardCharsets.UTF_8));
            digest[6] = (byte) ((digest[6] & 0x0f) | 0x50);
            digest[8] = (byte) ((digest[8] & 0x3f) | 0x80);
            java.nio.ByteBuffer output = java.nio.ByteBuffer.wrap(digest);
            return new UUID(output.getLong(), output.getLong()).toString();
        } catch (java.security.NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }
}
