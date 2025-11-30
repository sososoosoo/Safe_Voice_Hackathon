import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import '../services/audio_service.dart';
import '../styles.dart'; // 스타일 파일 임포트

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final AudioService _audioService = AudioService();
  bool _isRecording = false;
  bool _isAnalyzing = false;
  String _status = '안전 모니터링 준비중';

  // 기존 로직은 그대로 유지
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(gradient: gAppBg),
        child: SafeArea(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // --- 헤더 ---
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(vertical: 28, horizontal: 24),
                    decoration: BoxDecoration(
                      gradient: gHero,
                      borderRadius: BorderRadius.circular(28),
                      boxShadow: const [shadowLg],
                    ),
                    child: const Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Safe Voice',
                          style: TextStyle(fontSize: 34, fontWeight: FontWeight.w800, color: Colors.white, letterSpacing: 0.2),
                        ),
                        SizedBox(height: 6),
                        Text(
                          '여성 안심 귀갓길',
                          style: TextStyle(fontSize: 16, color: Color.fromRGBO(255, 255, 255, 0.92)),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),

                  // --- 상태 카드 ---
                  Container(
                    margin: const EdgeInsets.symmetric(horizontal: 6),
                    padding: const EdgeInsets.all(1.0),
                    decoration: BoxDecoration(
                      gradient: gBorder,
                      borderRadius: BorderRadius.circular(28),
                      boxShadow: const [shadowMd],
                    ),
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 20),
                      decoration: BoxDecoration(
                        gradient: gCard,
                        borderRadius: BorderRadius.circular(27),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Container(
                                width: 10, height: 10,
                                decoration: BoxDecoration(color: _getStatusColor(), shape: BoxShape.circle),
                              ),
                              const SizedBox(width: 10),
                              const Text('현재 상태', style: TextStyle(color: ink, fontWeight: FontWeight.w600)),
                            ],
                          ),
                          const SizedBox(height: 10),
                          Text(_status, style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w900, color: ink)),
                          const SizedBox(height: 4),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              const Text('현재 위치: 강원특별자치도 강릉시 해안로 536', style: TextStyle(color: muted, fontSize: 15)),
                              GestureDetector(
                                onTap: () { /* 위치 새로고침 기능 (현재는 비어있음) */ },
                                child: const Text('위치 새로고침', style: TextStyle(color: Color(0xFF2563EB), fontWeight: FontWeight.w800)),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 18),

                  // --- 마이크 버튼 ---
                  if (_isAnalyzing)
                    const SizedBox(
                      height: 200,
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            SpinKitWave(color: Colors.pink, size: 50.0),
                            SizedBox(height: 20),
                            Text('음성 분석 중...', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500, color: Colors.pink)),
                          ],
                        ),
                      ),
                    )
                  else
                    Center(
                      child: Column(
                        children: [
                          GestureDetector(
                            onTap: _toggleRecording,
                            child: Container(
                              width: 152, height: 152,
                              decoration: BoxDecoration(
                                gradient: gMic,
                                shape: BoxShape.circle,
                                boxShadow: [
                                  const BoxShadow(color: Color.fromRGBO(0, 0, 0, 0.28), blurRadius: 46, offset: Offset(0, 26)),
                                  BoxShadow(color: Colors.white.withOpacity(0.05), spreadRadius: 1, blurRadius: 0),
                                ],
                              ),
                              child: const Icon(Icons.mic, color: Colors.white, size: 48),
                            ),
                          ),
                          const SizedBox(height: 14),
                          Text(
                            _isRecording ? '모니터링 중' : '감지 시작',
                            style: const TextStyle(fontSize: 28, fontWeight: FontWeight.w900, color: ink),
                          ),
                        ],
                      ),
                    ),
                  const SizedBox(height: 6),

                  // --- 긴급 신고 ---
                  const Padding(
                    padding: EdgeInsets.fromLTRB(8, 20, 8, 12),
                    child: Text('긴급 신고', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w900, color: ink)),
                  ),
                  GestureDetector(
                    onTap: () { /* 112 신고 기능 (현재는 비어있음) */ },
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 20),
                      decoration: BoxDecoration(
                        gradient: gSos,
                        borderRadius: BorderRadius.circular(18),
                        boxShadow: const [shadowMd],
                      ),
                      child: const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.phone, color: Colors.white, size: 24),
                          SizedBox(width: 10),
                          Text('경찰서 112', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w900, color: Colors.white)),
                        ],
                      ),
                    ),
                  ),

                  // --- 빠른 액션 ---
                  const Padding(
                    padding: EdgeInsets.fromLTRB(8, 20, 8, 12),
                    child: Text('빠른 액션', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w900, color: ink)),
                  ),
                  GridView.count(
                    crossAxisCount: 2,
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    crossAxisSpacing: 12,
                    mainAxisSpacing: 12,
                    childAspectRatio: 2.2, // 타일 비율 조정
                    children: [
                      _buildActionTile(gBlue, Icons.location_on, '내 위치 공유'),
                      _buildActionTile(gGreen, Icons.family_restroom, '가족에게 연락'),
                      _buildActionTile(gPurple, Icons.chat_bubble, '도움 요청'),
                      _buildActionTile(gIndigo, Icons.directions_walk, '귀가 모드'),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  // 빠른 액션 타일 위젯 빌더
  Widget _buildActionTile(Gradient gradient, IconData icon, String label) {
    return GestureDetector(
      onTap: () { /* 각 액션 기능 (현재는 비어있음) */ },
      child: Container(
        decoration: BoxDecoration(
          gradient: gradient,
          borderRadius: BorderRadius.circular(14),
          boxShadow: const [shadowSm],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: Colors.white, size: 22),
            const SizedBox(height: 8),
            Text(label, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w800, fontSize: 16)),
          ],
        ),
      ),
    );
  }

  // --- 기존 상태 관리 및 로직은 모두 그대로 유지 ---

  Color _getStatusColor() {
    if (_isRecording) return Colors.red;
    if (_isAnalyzing) return Colors.orange;
    if (_status.contains('위험 상황 감지')) return Colors.red;
    if (_status.contains('정상')) return Colors.green;
    return const Color(0xFF3B82F6); // 기본 파란색
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      await _stopRecording();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    setState(() {
      _isRecording = true;
      _status = '음성 모니터링 중...';
    });

    try {
      await _audioService.startRecording();
      await Future.delayed(const Duration(seconds: 5));
      if (_isRecording) {
        await _stopRecording();
      }
    } catch (e) {
      setState(() {
        _isRecording = false;
        _status = '녹음 실패: $e';
      });
    }
  }

  Future<void> _stopRecording() async {
    setState(() {
      _isRecording = false;
      _isAnalyzing = true;
      _status = '음성 분석 중...';
    });

    try {
      final result = await _audioService.stopRecordingAndAnalyze();
      setState(() {
        _isAnalyzing = false;
        if (result['is_scream'] == true) {
          _status = '🚨 위험 상황 감지! 긴급 신고됨';
        } else {
          _status = '✅ 정상 - 위험 상황 없음';
        }
      });

      if (result['is_scream'] == true) {
        _showEmergencyAlert();
      }

      await Future.delayed(const Duration(seconds: 3));
      if (mounted) {
        setState(() {
          _status = '안전 모니터링 준비됨';
        });
      }
    } catch (e) {
      setState(() {
        _isAnalyzing = false;
        _status = '분석 실패: $e';
      });

      await Future.delayed(const Duration(seconds: 3));
      if (mounted) {
        setState(() {
          _status = '안전 모니터링 준비중';
        });
      }
    }
  }

  void _showEmergencyAlert() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Row(
          children: [
            Icon(Icons.warning, color: Colors.red, size: 30),
            SizedBox(width: 10),
            Text('긴급 상황 감지', style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold)),
          ],
        ),
        content: const Text(
          '비명 소리가 감지되었습니다.\n\n✓ 긴급 신고가 전송되었습니다\n✓ 현재 위치가 공유되었습니다',
          style: TextStyle(fontSize: 16, height: 1.5),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            style: TextButton.styleFrom(
              backgroundColor: Colors.pink,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
            child: const Padding(
              padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              child: Text('확인'),
            ),
          ),
        ],
      ),
    );
  }
}