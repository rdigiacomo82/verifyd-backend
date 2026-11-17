# VeriFYD Circle — Android native production project source

This directory now includes a **Gradle Android application module** around
the existing native FCM + TWA + native WebRTC source.

- Package **com.verifyd.circle**, never change to Detector com.verifyd.app
- Android compile/target API 36, minimum 24
- Gradle 8.13+ / JDK 17 / Android Gradle Plugin 8.13.2
- Firebase BoM 35.2.0, Play Integrity 1.6.0, WebRTC Android 150.7871.04
- Google Play **App Signing** SHA-256 in Digital Asset Links is mandatory
- No signer, keystore, google-services.json, or .aab is checked in
- This newly assembled project is **not** the original signed Windows project;
  import it into the existing project's app module without overwriting its
  signing config, version codes, icons, names, account or Play identity.
- Source requires a real Circle Firebase `google-services.json`, supplied
  privately at `app/google-services.json`.
- Provision the correct linked Google Cloud project number through private
  Gradle property `CIRCLE_PLAY_INTEGRITY_CLOUD_PROJECT_NUMBER`. A placeholder
  deliberately fails attestation at runtime.
- Do not build or distribute as production until verified `vfvid.com` DAL,
  Android notifications, FCM, App Integrity, native media, and signed-app
  association exist. No local signing keys are needed for GitHub source work.

### Source integration
This is a full standalone Gradle app module while original Windows app source
remains unmounted. It wires all staged native Java files directly using
Gradle `sourceSets.main.java.srcDirs` and the existing manifest fragment.

Development APKs cannot pass production Play Integrity: install through the
proper Play track with the **original Play signing identity**. No automatic
release, backend activation or Play Console upload is performed.
