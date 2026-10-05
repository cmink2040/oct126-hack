from chiro import guardrails


def test_red_flags_detected_in_generated_text():
    from chiro.datagen import RED_FLAG_TEXT

    for text in RED_FLAG_TEXT:
        assert guardrails.detect_red_flags(text), text


def test_ordinary_complaints_are_not_red_flags():
    assert guardrails.detect_red_flags("my lower back has been killing me for two weeks") == []


def test_banned_claims_rejected():
    problems = guardrails.check_message("Our adjustments cure sciatica, guaranteed!", "email")
    assert any("cure" in p for p in problems) and any("guarantee" in p for p in problems)


def test_word_boundaries_avoid_false_positives():
    assert guardrails.check_message("We can secure you a spot on Tuesday at 9am with Dr. Patel.", "email") == []


def test_sms_length_and_opt_out():
    assert guardrails.check_message("x" * 400, "sms")
    msg = guardrails.ensure_sms_opt_out("Hi Sam, we have Tue 9am open - want it?")
    assert msg.endswith(guardrails.SMS_OPT_OUT)
    assert guardrails.ensure_sms_opt_out(msg) == msg
