# ============================================================
# VeriFYD Circle — localized Trust Voice push copy
# VERIFYD_TRUST_VOICE_PUSH_I18N_V1
# ============================================================

from __future__ import annotations

from typing import Any

DEFAULT_LOCALE = "en-US"

MESSAGES = {
    "en-US": {
        "message_title": "New Trust Voice message",
        "message_body": "{sender} sent you a message.",
        "video_title": "Video meeting started",
        "video_body": "{starter} started a video meeting in {group}.",
        "scheduled_new": "New meeting scheduled",
        "scheduled_updated": "Meeting updated",
        "scheduled_cancelled": "Meeting cancelled",
        "scheduled_body_new": "{title} was scheduled in {group}.",
        "scheduled_body_updated": "{title} was updated in {group}.",
        "scheduled_body_cancelled": "{title} was cancelled in {group}.",
        "reminder_title": "Meeting starts in 15 minutes",
        "results_title": "Meeting results ready",
        "results_recording": "{group} summary and recording are ready.",
        "results_summary": "{group} meeting summary is ready.",
        "test_title": "Trust Voice notifications",
        "test_body": "Push notifications are working on this device.",
        "someone": "Someone",
        "your_group": "your group",
        "meeting": "Trust Voice meeting",
    },
    "es-MX": {
        "message_title": "Nuevo mensaje de Trust Voice",
        "message_body": "{sender} te envió un mensaje.",
        "video_title": "La reunión de video comenzó",
        "video_body": "{starter} inició una reunión de video en {group}.",
        "scheduled_new": "Nueva reunión programada",
        "scheduled_updated": "Reunión actualizada",
        "scheduled_cancelled": "Reunión cancelada",
        "scheduled_body_new": "{title} se programó en {group}.",
        "scheduled_body_updated": "{title} se actualizó en {group}.",
        "scheduled_body_cancelled": "{title} se canceló en {group}.",
        "reminder_title": "La reunión comienza en 15 minutos",
        "results_title": "Resultados de la reunión listos",
        "results_recording": "El resumen y la grabación de {group} están listos.",
        "results_summary": "El resumen de la reunión de {group} está listo.",
        "test_title": "Notificaciones de Trust Voice",
        "test_body": "Las notificaciones push funcionan en este dispositivo.",
        "someone": "Alguien",
        "your_group": "tu grupo",
        "meeting": "Reunión de Trust Voice",
    },
    "fr-FR": {
        "message_title": "Nouveau message Trust Voice",
        "message_body": "{sender} vous a envoyé un message.",
        "video_title": "Réunion vidéo démarrée",
        "video_body": "{starter} a démarré une réunion vidéo dans {group}.",
        "scheduled_new": "Nouvelle réunion planifiée",
        "scheduled_updated": "Réunion mise à jour",
        "scheduled_cancelled": "Réunion annulée",
        "scheduled_body_new": "{title} a été planifiée dans {group}.",
        "scheduled_body_updated": "{title} a été mise à jour dans {group}.",
        "scheduled_body_cancelled": "{title} a été annulée dans {group}.",
        "reminder_title": "La réunion commence dans 15 minutes",
        "results_title": "Résultats de la réunion disponibles",
        "results_recording": "Le résumé et l’enregistrement de {group} sont prêts.",
        "results_summary": "Le résumé de la réunion {group} est prêt.",
        "test_title": "Notifications Trust Voice",
        "test_body": "Les notifications push fonctionnent sur cet appareil.",
        "someone": "Quelqu’un",
        "your_group": "votre groupe",
        "meeting": "Réunion Trust Voice",
    },
    "pt-BR": {
        "message_title": "Nova mensagem do Trust Voice",
        "message_body": "{sender} enviou uma mensagem para você.",
        "video_title": "Reunião de vídeo iniciada",
        "video_body": "{starter} iniciou uma reunião de vídeo em {group}.",
        "scheduled_new": "Nova reunião agendada",
        "scheduled_updated": "Reunião atualizada",
        "scheduled_cancelled": "Reunião cancelada",
        "scheduled_body_new": "{title} foi agendada em {group}.",
        "scheduled_body_updated": "{title} foi atualizada em {group}.",
        "scheduled_body_cancelled": "{title} foi cancelada em {group}.",
        "reminder_title": "A reunião começa em 15 minutos",
        "results_title": "Resultados da reunião prontos",
        "results_recording": "O resumo e a gravação de {group} estão prontos.",
        "results_summary": "O resumo da reunião de {group} está pronto.",
        "test_title": "Notificações do Trust Voice",
        "test_body": "As notificações push estão funcionando neste dispositivo.",
        "someone": "Alguém",
        "your_group": "seu grupo",
        "meeting": "Reunião do Trust Voice",
    },
    "ar": {
        "message_title": "رسالة جديدة من Trust Voice",
        "message_body": "أرسل لك {sender} رسالة.",
        "video_title": "بدأ اجتماع الفيديو",
        "video_body": "بدأ {starter} اجتماع فيديو في {group}.",
        "scheduled_new": "تمت جدولة اجتماع جديد",
        "scheduled_updated": "تم تحديث الاجتماع",
        "scheduled_cancelled": "تم إلغاء الاجتماع",
        "scheduled_body_new": "تمت جدولة {title} في {group}.",
        "scheduled_body_updated": "تم تحديث {title} في {group}.",
        "scheduled_body_cancelled": "تم إلغاء {title} في {group}.",
        "reminder_title": "يبدأ الاجتماع خلال 15 دقيقة",
        "results_title": "نتائج الاجتماع جاهزة",
        "results_recording": "ملخص وتسجيل {group} جاهزان.",
        "results_summary": "ملخص اجتماع {group} جاهز.",
        "test_title": "إشعارات Trust Voice",
        "test_body": "إشعارات الدفع تعمل على هذا الجهاز.",
        "someone": "شخص ما",
        "your_group": "مجموعتك",
        "meeting": "اجتماع Trust Voice",
    },
    "hi-IN": {
        "message_title": "नया Trust Voice संदेश",
        "message_body": "{sender} ने आपको संदेश भेजा।",
        "video_title": "वीडियो मीटिंग शुरू हुई",
        "video_body": "{starter} ने {group} में वीडियो मीटिंग शुरू की।",
        "scheduled_new": "नई मीटिंग शेड्यूल हुई",
        "scheduled_updated": "मीटिंग अपडेट हुई",
        "scheduled_cancelled": "मीटिंग रद्द हुई",
        "scheduled_body_new": "{title} को {group} में शेड्यूल किया गया।",
        "scheduled_body_updated": "{title} को {group} में अपडेट किया गया।",
        "scheduled_body_cancelled": "{title} को {group} में रद्द किया गया।",
        "reminder_title": "मीटिंग 15 मिनट में शुरू होगी",
        "results_title": "मीटिंग के परिणाम तैयार हैं",
        "results_recording": "{group} का सारांश और रिकॉर्डिंग तैयार हैं।",
        "results_summary": "{group} मीटिंग का सारांश तैयार है।",
        "test_title": "Trust Voice सूचनाएँ",
        "test_body": "इस डिवाइस पर पुश सूचनाएँ काम कर रही हैं।",
        "someone": "कोई व्यक्ति",
        "your_group": "आपका समूह",
        "meeting": "Trust Voice मीटिंग",
    },
    "zh-CN": {
        "message_title": "新的 Trust Voice 消息",
        "message_body": "{sender} 给你发了一条消息。",
        "video_title": "视频会议已开始",
        "video_body": "{starter} 在 {group} 中发起了视频会议。",
        "scheduled_new": "已安排新会议",
        "scheduled_updated": "会议已更新",
        "scheduled_cancelled": "会议已取消",
        "scheduled_body_new": "已在 {group} 中安排 {title}。",
        "scheduled_body_updated": "已在 {group} 中更新 {title}。",
        "scheduled_body_cancelled": "已在 {group} 中取消 {title}。",
        "reminder_title": "会议将在 15 分钟后开始",
        "results_title": "会议结果已准备好",
        "results_recording": "{group} 的摘要和录制内容已准备好。",
        "results_summary": "{group} 的会议摘要已准备好。",
        "test_title": "Trust Voice 通知",
        "test_body": "此设备的推送通知工作正常。",
        "someone": "某人",
        "your_group": "你的群组",
        "meeting": "Trust Voice 会议",
    },
    "zh-TW": {
        "message_title": "新的 Trust Voice 訊息",
        "message_body": "{sender} 傳送了一則訊息給你。",
        "video_title": "視訊會議已開始",
        "video_body": "{starter} 在 {group} 中發起了視訊會議。",
        "scheduled_new": "已安排新會議",
        "scheduled_updated": "會議已更新",
        "scheduled_cancelled": "會議已取消",
        "scheduled_body_new": "已在 {group} 中安排 {title}。",
        "scheduled_body_updated": "已在 {group} 中更新 {title}。",
        "scheduled_body_cancelled": "已在 {group} 中取消 {title}。",
        "reminder_title": "會議將在 15 分鐘後開始",
        "results_title": "會議結果已準備好",
        "results_recording": "{group} 的摘要與錄製內容已準備好。",
        "results_summary": "{group} 的會議摘要已準備好。",
        "test_title": "Trust Voice 通知",
        "test_body": "此裝置的推播通知運作正常。",
        "someone": "某人",
        "your_group": "你的群組",
        "meeting": "Trust Voice 會議",
    },
    "ja-JP": {
        "message_title": "新しい Trust Voice メッセージ",
        "message_body": "{sender} からメッセージが届きました。",
        "video_title": "ビデオミーティングが開始されました",
        "video_body": "{starter} が {group} でビデオミーティングを開始しました。",
        "scheduled_new": "新しいミーティングが予定されました",
        "scheduled_updated": "ミーティングが更新されました",
        "scheduled_cancelled": "ミーティングがキャンセルされました",
        "scheduled_body_new": "{title} が {group} に予定されました。",
        "scheduled_body_updated": "{title} が {group} で更新されました。",
        "scheduled_body_cancelled": "{title} が {group} でキャンセルされました。",
        "reminder_title": "ミーティングは15分後に開始します",
        "results_title": "ミーティング結果の準備ができました",
        "results_recording": "{group} の要約と録画の準備ができました。",
        "results_summary": "{group} のミーティング要約の準備ができました。",
        "test_title": "Trust Voice 通知",
        "test_body": "この端末でプッシュ通知が正常に動作しています。",
        "someone": "誰か",
        "your_group": "あなたのグループ",
        "meeting": "Trust Voice ミーティング",
    },
}


def normalize_locale(locale: str) -> str:
    value = str(locale or "").strip()
    return value if value in MESSAGES else DEFAULT_LOCALE


def push_text(locale: str, key: str, **values: Any) -> str:
    normalized = normalize_locale(locale)
    template = MESSAGES.get(normalized, {}).get(key) or MESSAGES[DEFAULT_LOCALE].get(key) or key
    try:
        return str(template).format(**values)
    except Exception:
        return str(template)


def localize_notification(
    locale: str,
    notification_type: str,
    title: str,
    body: str,
) -> tuple[str, str]:
    """Translate privacy-safe push copy while preserving dynamic names/titles."""
    kind = str(notification_type or "").strip()
    current_title = str(title or "")
    current_body = str(body or "")

    if kind == "message_created":
        suffix = " sent you a message."
        sender = current_body[:-len(suffix)] if current_body.endswith(suffix) else push_text(locale, "someone")
        return push_text(locale, "message_title"), push_text(locale, "message_body", sender=sender)

    if kind == "group_video_started":
        marker = " started a video meeting in "
        if marker in current_body and current_body.endswith("."):
            starter, group = current_body[:-1].split(marker, 1)
        else:
            starter, group = push_text(locale, "someone"), push_text(locale, "your_group")
        return push_text(locale, "video_title"), push_text(locale, "video_body", starter=starter, group=group)

    scheduled = {
        "group_video_schedule_created": ("scheduled_new", "scheduled_body_new", " was scheduled in "),
        "group_video_schedule_updated": ("scheduled_updated", "scheduled_body_updated", " was updated in "),
        "group_video_schedule_cancelled": ("scheduled_cancelled", "scheduled_body_cancelled", " was cancelled in "),
    }
    if kind in scheduled:
        title_key, body_key, marker = scheduled[kind]
        if marker in current_body and current_body.endswith("."):
            meeting_title, group = current_body[:-1].split(marker, 1)
        else:
            meeting_title, group = push_text(locale, "meeting"), push_text(locale, "your_group")
        return push_text(locale, title_key), push_text(locale, body_key, title=meeting_title, group=group)

    if kind == "scheduled_meeting_reminder":
        return push_text(locale, "reminder_title"), current_body

    if kind == "meeting_results_ready":
        recording_suffix = " summary and recording are ready."
        summary_suffix = " meeting summary is ready."
        if current_body.endswith(recording_suffix):
            group = current_body[:-len(recording_suffix)]
            return push_text(locale, "results_title"), push_text(locale, "results_recording", group=group)
        if current_body.endswith(summary_suffix):
            group = current_body[:-len(summary_suffix)]
            return push_text(locale, "results_title"), push_text(locale, "results_summary", group=group)
        return push_text(locale, "results_title"), current_body

    if kind == "push_test":
        return push_text(locale, "test_title"), push_text(locale, "test_body")

    return current_title, current_body
