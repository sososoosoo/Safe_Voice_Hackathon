import 'package:flutter/material.dart';

// HTML CSS :root 변수를 Flutter 상수로 변환

// 기본 색상
const Color ink = Color(0xFF0B0B0C);
const Color muted = Color(0xFF6B7280);

// 그라데이션
const Gradient gAppBg = RadialGradient(
  center: Alignment(-1.5, -1.2),
  radius: 5.5,
  colors: [Color(0xFFF1F5F9), Color(0xFFEEF2F7), Color(0xFFE9EDF5)],
  stops: [0.0, 0.4, 1.0],
);

const Gradient gHero = LinearGradient(
  begin: Alignment(0.8, -0.6),
  end: Alignment(-0.8, 0.6),
  colors: [Color(0xFF0E0F11), Color(0xFF15161A), Color(0xFF0C0C0E)],
);

const Gradient gCard = LinearGradient(
  begin: Alignment.topCenter,
  end: Alignment.bottomCenter,
  colors: [Color(0xFFFFFFFF), Color(0xFFFAFAFA)],
);

const Gradient gBorder = LinearGradient(
  begin: Alignment.topCenter,
  end: Alignment.bottomCenter,
  colors: [Color.fromRGBO(0, 0, 0, 0.16), Color.fromRGBO(255, 255, 255, 0.38)],
);

const Gradient gMic = LinearGradient(
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
  colors: [Color(0xFF666666), Color(0xFF222222)],
);

const Gradient gSos = LinearGradient(
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
  colors: [Color(0xFFF43F5E), Color(0xFFEF4444), Color(0xFFF97316)],
  stops: [0.0, 0.6, 1.15],
);

const Gradient gBlue = LinearGradient(
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
  colors: [Color(0xFF60A5FA), Color(0xFF3B82F6)],
);

const Gradient gGreen = LinearGradient(
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
  colors: [Color(0xFF34D399), Color(0xFF16A34A)],
);

const Gradient gPurple = LinearGradient(
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
  colors: [Color(0xFFA78BFA), Color(0xFF8B5CF6)],
);

const Gradient gIndigo = LinearGradient(
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
  colors: [Color(0xFF818CF8), Color(0xFF6366F1)],
);

// 그림자
const BoxShadow shadowLg = BoxShadow(
  color: Color.fromRGBO(0, 0, 0, 0.18),
  blurRadius: 48,
  offset: Offset(0, 24),
);

const BoxShadow shadowMd = BoxShadow(
  color: Color.fromRGBO(0, 0, 0, 0.12),
  blurRadius: 32,
  offset: Offset(0, 14),
);

const BoxShadow shadowSm = BoxShadow(
  color: Color.fromRGBO(0, 0, 0, 0.10),
  blurRadius: 18,
  offset: Offset(0, 8),
);
