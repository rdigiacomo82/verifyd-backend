package com.verifyd.circle.calls;

import android.content.Context;
import android.content.res.Resources;
import android.util.Base64;
import com.google.android.gms.tasks.Tasks;
import com.google.android.play.core.integrity.IntegrityManagerFactory;
import com.google.android.play.core.integrity.StandardIntegrityManager;
import com.google.android.play.core.integrity.StandardIntegrityManager.PrepareIntegrityTokenRequest;
import com.google.android.play.core.integrity.StandardIntegrityManager.StandardIntegrityTokenRequest;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.concurrent.TimeUnit;

/**
 * Signed Circle production app attestation. Used ONLY on a background worker.
 * Fail closed when the app/Cloud Play Integrity project has not been paired.
 *
 * Supply integer resource circle_play_integrity_cloud_project_number using
 * the actual Google Cloud project linked to com.verifyd.circle in Play Console.
 * Do not reuse the Detector pilot project number without verifying the linkage.
 */
final class CirclePlayIntegrity {
    private CirclePlayIntegrity() {}

    static String attest(Context context, String canonicalProof, String fcmToken) throws Exception {
        if (!"com.verifyd.circle".equals(context.getPackageName()))
            throw new SecurityException("Not the signed production Circle package");
        int resource = context.getResources().getIdentifier(
                "circle_play_integrity_cloud_project_number", "string", context.getPackageName());
        if (resource == 0) throw new IllegalStateException("Play Integrity project not configured");
        String value = context.getString(resource).trim();
        if (!value.matches("[0-9]{6,20}"))
            throw new IllegalStateException("Invalid Play Integrity project number");
        long cloudProject = Long.parseLong(value);

        byte[] fcmDigest = MessageDigest.getInstance("SHA-256")
                .digest(fcmToken.getBytes(StandardCharsets.US_ASCII));
        StringBuilder fcmHex = new StringBuilder(64);
        for (byte b : fcmDigest) fcmHex.append(String.format(java.util.Locale.ROOT, "%02x", b & 0xff));
        byte[] hash = MessageDigest.getInstance("SHA-256")
                .digest((canonicalProof + "\n" + fcmHex).getBytes(StandardCharsets.US_ASCII));
        String requestHash = Base64.encodeToString(
                hash, Base64.URL_SAFE | Base64.NO_WRAP | Base64.NO_PADDING);

        StandardIntegrityManager manager = IntegrityManagerFactory.createStandard(context);
        StandardIntegrityManager.StandardIntegrityTokenProvider provider =
                Tasks.await(manager.prepareIntegrityToken(
                        PrepareIntegrityTokenRequest.builder()
                                .setCloudProjectNumber(cloudProject).build()),
                        30, TimeUnit.SECONDS);
        String token = Tasks.await(provider.request(
                        StandardIntegrityTokenRequest.builder().setRequestHash(requestHash).build()),
                        20, TimeUnit.SECONDS).token();
        if (token == null || token.length() < 100)
            throw new SecurityException("Play Integrity token unavailable");
        return token;
    }
}
