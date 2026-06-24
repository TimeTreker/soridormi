# Chromie Bilingual OuteTTS Speaker Guide

This guide creates a Chromie OuteTTS speaker profile that works better for both
English and Chinese. OuteTTS can synthesize both languages, but the speaker
reference strongly affects accent, tone, and pronunciation. A clean bilingual
reference WAV is usually better than the built-in English-only test speaker.

## Goal

Create this file in the Chromie repo:

```text
/home/chromie/github/chromie/tts/speakers/chromie_bilingual_zh_en.wav
```

Then convert it into this reusable OuteTTS speaker profile:

```text
/home/chromie/github/chromie/tts/speakers/chromie_bilingual_zh_en.json
```

## Recording Setup

Use a quiet room, one speaker, one microphone, and no background music. Keep the
microphone 15-25 cm from your mouth. Speak naturally, with steady volume, and
leave about half a second of silence at the start and end.

Recommended recording target:

- Format: WAV
- Channels: mono is best, stereo is acceptable
- Sample rate: 44.1 kHz or 48 kHz
- Duration: 25-45 seconds
- Volume: loud enough to be clear, but never clipping
- Content: one continuous bilingual reading from the script below

Avoid:

- phone call audio, Bluetooth headset audio, echo, fans, keyboard noise;
- whispering, shouting, acting, laughing during the sample;
- heavy reverb or noise reduction artifacts;
- switching between different speakers.

## Recording With A GUI

Audacity is the easiest path:

1. Open Audacity.
2. Select the real microphone input.
3. Record the script below in one take.
4. Trim long silence at the beginning and end.
5. Export as WAV.
6. Save it as:

```text
/home/chromie/github/chromie/tts/speakers/chromie_bilingual_zh_en.wav
```

## Recording From The Terminal

First list audio devices:

```bash
wpctl status
```

Record with PipeWire:

```bash
cd /home/chromie/github/chromie
mkdir -p tts/speakers
pw-record --rate 48000 --channels 1 tts/speakers/chromie_bilingual_zh_en.wav
```

Read the script, then press `Ctrl+C` to stop recording.

If `pw-record` is not available, use any recorder that can export WAV and place
the final file at:

```text
/home/chromie/github/chromie/tts/speakers/chromie_bilingual_zh_en.wav
```

## Bilingual Reference Script

Read this naturally. Do not rush. Keep the same voice and distance from the
microphone for both languages.

```text
你好，我是 Chromie。今天我会用中文和英文说话。
我可以清楚地说普通话，也可以自然地说 English.
现在我们测试中文的声调、节奏、停顿，还有轻声。
请向前走十秒，然后点头两次，最后看着我。
如果你听得见我，请说：准备好了，我们开始吧。

Hello, I am Chromie. I can speak naturally in English and Chinese.
Today we are testing a warm, clear, friendly robot voice.
Please walk forward for ten seconds, then nod twice, then look at me.
The voice should sound calm, helpful, and easy to understand.
Now the bilingual speaker profile is ready for everyday conversation.
```

For an even better profile, record a second take and choose the cleaner one. The
best take is not the most dramatic one; it is the clearest, steadiest one.

## Create The Speaker Profile

Start Chromie's services first if they are not already running:

```bash
cd /home/chromie/github/chromie
./scripts/start_services.sh
```

Create the OuteTTS speaker profile and make it the default:

```bash
cd /home/chromie/github/chromie
./scripts/create_speaker_in_container.sh \
  /app/speakers/chromie_bilingual_zh_en.wav \
  chromie_bilingual_zh_en \
  --make-default
```

This should create:

```text
tts/speakers/chromie_bilingual_zh_en.json
tts/speakers/default.json
```

Chromie's orchestrator normally uses `TTS_SPEAKER_ID=default`, so making this
speaker the default is the simplest setup.

## Test English And Chinese

Use Chromie's text-to-MuJoCo voice path with speaker output:

```bash
cd /home/chromie/github/chromie
./scripts/run_voice_mujoco_text_case.sh "Hello, I am Chromie. I can speak English clearly." --speaker
./scripts/run_voice_mujoco_text_case.sh "你好，我是 Chromie。我可以自然地说中文。" --speaker
```

If the speaker is still strange in Chinese, make a new reference WAV with:

- slightly slower Chinese;
- cleaner room audio;
- less English accent while reading Chinese;
- no clipped peaks;
- no long silence in the middle.

Then rerun the speaker creation command with the same `speaker_id`.

## Quality Checklist

Before accepting the speaker profile, confirm:

- English words are understandable and stable.
- Chinese tones are not flat or random.
- The voice does not sound like two different people.
- Short mixed-language sentences sound acceptable.
- The robot voice is calm enough for repeated interaction.

Recommended final test sentence:

```text
你好，我是 Chromie. I can help you in English and Chinese.
```

## Notes

OuteTTS may preserve the accent and style of the reference speaker. For best
Chinese-English results, the reference speaker should genuinely pronounce both
languages well. If one bilingual profile is not good enough, keep two profiles:

```text
chromie_zh
chromie_en
```

Then route Chinese replies to `chromie_zh` and English replies to `chromie_en`
in Chromie later.
