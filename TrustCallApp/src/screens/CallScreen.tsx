import React, { useEffect, useRef, useState } from 'react';
import {
  Alert,
  PermissionsAndroid,
  Platform,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  mediaDevices,
  RTCPeerConnection,
  RTCSessionDescription,
} from 'react-native-webrtc';
import {
  getBackendHttpUrl,
  getBackendWsUrl,
  getResolvedBackendBaseUrl,
} from '../config/backend';

type VerdictTone = 'safe' | 'warning' | 'danger' | 'neutral';

const CallScreen = ({ navigation, route }: any) => {
  const [callDuration, setCallDuration] = useState(0);
  const [localStream, setLocalStream] = useState<any>(null);
  const [peerConnection, setPeerConnection] = useState<any>(null);

  const [signalScore, setSignalScore] = useState<string>('Listening...');
  const [signalThreat, setSignalThreat] = useState<boolean>(false);
  const [semanticStatus, setSemanticStatus] = useState<string>('Waiting for speech');
  const [identityStatus, setIdentityStatus] = useState<string>('Checking caller');
  const [identityConfidence, setIdentityConfidence] = useState<string>('Pending');
  const [, setIdentityMode] = useState<string>('verification');
  const [, setIdentifiedCaller] = useState<string>('No candidate');
  const [, setIdentityChunks] = useState<number>(0);
  const [, setIdentityReason] = useState<string>('Waiting for live audio');
  const [audioFrames, setAudioFrames] = useState<number>(0);
  const [, setBufferedSeconds] = useState<string>('0.00s');
  const [fusionStatus, setFusionStatus] = useState<string>('WAITING');
  const [sessionId, setSessionId] = useState<string>('Not Connected');
  const [telemetryStatus, setTelemetryStatus] = useState<string>('Disconnected');
  const [candidateEmbeddingCount, setCandidateEmbeddingCount] = useState<number>(0);
  const [candidateStatus, setCandidateStatus] = useState<string>('Idle');
  const [candidateEnrollmentReady, setCandidateEnrollmentReady] = useState<boolean>(false);
  const [isCallActive, setIsCallActive] = useState(false);

  const ws = useRef<WebSocket | null>(null);
  const callerName = route?.params?.callerName ?? 'Unknown Caller';
  const callerId = route?.params?.callerId ?? 'unknown';
  const phoneNumber = route?.params?.phoneNumber ?? '';
  const resolvedFromContacts = route?.params?.resolvedFromContacts === true;

  const formatPercent = (value: unknown): string => {
    if (value == null || value === '') return 'Pending';
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(1)}%` : String(value);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  const normalizeText = (value: string) => value.toLowerCase().replace(/[_-]+/g, ' ');

  const applyIdentityTelemetry = (payload: any, chunkCount?: number) => {
    const identityText =
      payload.identity_match ??
      payload.display_text ??
      payload.status ??
      'Checking caller';
    const confidence =
      payload.identity_match_confidence ??
      payload.match_confidence ??
      null;
    const mode =
      payload.identity_mode ??
      payload.identification_mode ??
      'verification';
    const identified =
      payload.identity_identified_caller_id ??
      payload.identified_caller_id ??
      'No candidate';
    const reason =
      payload.identity_reason ??
      payload.reason ??
      'No runtime issue reported';

    setIdentityStatus(String(identityText));
    setIdentityConfidence(formatPercent(confidence));
    setIdentityMode(String(mode));
    setIdentifiedCaller(String(identified));
    setIdentityReason(String(reason));
    if (typeof chunkCount === 'number') setIdentityChunks(chunkCount);
    if (typeof payload.candidate_embedding_count === 'number') {
      setCandidateEmbeddingCount(payload.candidate_embedding_count);
    }
    if (typeof payload.candidate_enrollment_ready === 'boolean') {
      setCandidateEnrollmentReady(payload.candidate_enrollment_ready);
      setCandidateStatus(payload.candidate_enrollment_ready ? 'Ready to save' : 'Idle');
    }
  };

  const applySessionSnapshot = (session: any) => {
    setTelemetryStatus((current) =>
      current === 'Connected' ? 'Connected' : 'Receiving results',
    );
    setAudioFrames(typeof session.frames_received === 'number' ? session.frames_received : 0);
    setBufferedSeconds(
      typeof session.buffered_duration_seconds === 'number'
        ? `${session.buffered_duration_seconds.toFixed(2)}s`
        : '0.00s',
    );
    setIdentityChunks(typeof session.chunks_processed === 'number' ? session.chunks_processed : 0);
    setCandidateEmbeddingCount(
      typeof session.candidate_embedding_count === 'number'
        ? session.candidate_embedding_count
        : 0,
    );
    setCandidateEnrollmentReady(session.candidate_enrollment_ready === true);
    setCandidateStatus(String(session.candidate_status ?? 'Idle'));

    if (session.last_signal_score) {
      setSignalScore(session.last_signal_score);
      setSignalThreat(session.last_signal_threat === true);
    }
    if (session.last_semantic_intent) setSemanticStatus(String(session.last_semantic_intent));
    if (session.last_fusion_status) setFusionStatus(String(session.last_fusion_status));

    if (session.last_identity_result) {
      applyIdentityTelemetry(session.last_identity_result, session.chunks_processed);
      return;
    }

    setIdentityReason(session.last_error ?? session.state ?? 'Waiting for live audio');
  };

  const startAudioStream = async () => {
    try {
      if (Platform.OS === 'android') {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          {
            title: 'Microphone Permission',
            message: 'Trust-Call needs microphone access to protect this call.',
            buttonNeutral: 'Ask Me Later',
            buttonNegative: 'Cancel',
            buttonPositive: 'OK',
          },
        );
        if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
          Alert.alert('Microphone Required', 'Enable microphone access to start protection.');
          setIsCallActive(false);
          return;
        }
      }

      const stream = await mediaDevices.getUserMedia({ audio: true, video: false });
      setLocalStream(stream);

      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      });
      setPeerConnection(pc);

      stream.getTracks().forEach((track: any) => {
        pc.addTrack(track, stream);
      });

      const offer = await pc.createOffer({});
      await pc.setLocalDescription(offer);

      const response = await fetch(getBackendHttpUrl('/offer'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
          caller_id: callerId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const answer = await response.json();
      setSessionId(answer.session_id ?? 'Unknown Session');

      await pc.setRemoteDescription(
        new RTCSessionDescription({
          sdp: answer.sdp,
          type: answer.type,
        }),
      );

      ws.current = new WebSocket(getBackendWsUrl('/ws'));
      ws.current.onopen = () => setTelemetryStatus('Connected');
      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setTelemetryStatus('Connected');
          setSignalScore(data.signal_score);
          setSignalThreat(data.is_threat === true);
          setSemanticStatus(data.semantic_intent);
          applyIdentityTelemetry(data, data.identity_chunk_count);
          setFusionStatus(data.fusion_status);
        } catch (error) {
          console.log('Error parsing telemetry data', error);
        }
      };
      ws.current.onerror = () => setTelemetryStatus('Connection issue');
      ws.current.onclose = () => setTelemetryStatus('Disconnected');
    } catch (error) {
      console.log('Failed to start protected call:', error);
      setTelemetryStatus('Connection failed');
      Alert.alert(
        'Connection Failed',
        `Could not connect to Trust-Call at ${getResolvedBackendBaseUrl()}.`,
      );
      setIsCallActive(false);
    }
  };

  const handleAcceptCall = () => {
    setIsCallActive(true);
    setTelemetryStatus('Connecting');
    startAudioStream();
  };

  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    if (isCallActive) {
      timer = setInterval(() => setCallDuration((prev) => prev + 1), 1000);
    }
    return () => clearInterval(timer);
  }, [isCallActive]);

  useEffect(() => {
    if (!isCallActive || sessionId === 'Not Connected' || sessionId === 'Unknown Session') {
      return;
    }

    let cancelled = false;

    const pollSession = async () => {
      try {
        const response = await fetch(getBackendHttpUrl(`/identity/live/sessions/${sessionId}`));
        if (!response.ok) return;
        const session = await response.json();
        if (!cancelled) applySessionSnapshot(session);
      } catch (error) {
        console.log('Live session polling failed:', error);
      }
    };

    pollSession();
    const pollTimer = setInterval(pollSession, 1500);

    return () => {
      cancelled = true;
      clearInterval(pollTimer);
    };
    // applySessionSnapshot intentionally reads the latest telemetry setters only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isCallActive, sessionId]);

  const cleanupCallResources = () => {
    if (localStream) localStream.getTracks().forEach((track: any) => track.stop());
    if (peerConnection) peerConnection.close();
    if (ws.current) ws.current.close();
  };

  const discardCandidateProfile = async (currentSessionId: string) => {
    try {
      await fetch(getBackendHttpUrl(`/identity/live/sessions/${currentSessionId}/candidate`), {
        method: 'DELETE',
      });
    } catch (error) {
      console.log('Failed to discard TOFU candidate embeddings:', error);
    } finally {
      navigation.navigate('HomeScreen');
    }
  };

  const saveCandidateProfile = async (currentSessionId: string) => {
    try {
      const response = await fetch(
        getBackendHttpUrl(`/identity/live/sessions/${currentSessionId}/enroll-candidate`),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ safe_to_enroll: true }),
        },
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Failed to save voice profile.');
      }

      Alert.alert('Voice Profile Saved', `${callerName} is now protected by Trust-Call.`, [
        { text: 'OK', onPress: () => navigation.navigate('HomeScreen') },
      ]);
    } catch (error: any) {
      Alert.alert('Enrollment Failed', error?.message ?? 'Could not save this voice profile.', [
        { text: 'OK', onPress: () => navigation.navigate('HomeScreen') },
      ]);
    }
  };

  const handleEndCall = () => {
    const currentSessionId = sessionId;
    const hasCandidateProfile =
      candidateEnrollmentReady &&
      candidateEmbeddingCount > 0 &&
      currentSessionId !== 'Not Connected' &&
      currentSessionId !== 'Unknown Session';

    cleanupCallResources();
    setIsCallActive(false);

    if (!hasCandidateProfile) {
      navigation.navigate('HomeScreen');
      return;
    }

    Alert.alert(
      'Save Voice Profile?',
      `Trust-Call learned enough voice samples for ${callerName}. Save this as the trusted profile?`,
      [
        {
          text: 'Discard',
          style: 'destructive',
          onPress: () => discardCandidateProfile(currentSessionId),
        },
        {
          text: 'Save Profile',
          onPress: () => saveCandidateProfile(currentSessionId),
        },
      ],
    );
  };

  const isWaiting = fusionStatus === 'WAITING' || !isCallActive;
  const normalizedFusion = normalizeText(fusionStatus);
  const normalizedIdentity = normalizeText(identityStatus);
  const normalizedSemantic = normalizeText(semanticStatus);

  const overallTone: VerdictTone = signalThreat || normalizedFusion.includes('threat')
    ? 'danger'
    : normalizedFusion.includes('review') ||
        normalizedIdentity.includes('mismatch') ||
        normalizedSemantic.includes('coerc')
      ? 'warning'
      : isWaiting
        ? 'neutral'
        : 'safe';

  const overallTitle =
    overallTone === 'danger'
      ? 'Threat detected'
      : overallTone === 'warning'
        ? 'Review this call'
        : overallTone === 'safe'
          ? 'No immediate threat'
          : isCallActive
            ? 'Listening'
            : 'Ready to protect';

  const overallSubtitle =
    overallTone === 'danger'
      ? 'Trust-Call found a strong risk signal. End the call if this is unexpected.'
      : overallTone === 'warning'
        ? 'One auditor is unsure. Continue carefully and avoid sharing sensitive information.'
        : overallTone === 'safe'
          ? 'The latest checks did not find an urgent risk.'
          : isCallActive
            ? 'Speak naturally for a few seconds while Trust-Call analyzes the call.'
            : 'Accept the simulated call to begin live analysis.';

  const identitySubtitle =
    normalizedIdentity.includes('not enrolled')
      ? 'No trusted voice profile yet. You can save one after a safe call.'
      : normalizedIdentity.includes('match')
        ? `Caller voice matches the saved profile (${identityConfidence}).`
        : normalizedIdentity.includes('mismatch')
          ? `Caller voice does not match the saved profile (${identityConfidence}).`
          : 'Waiting for enough live speech.';

  const semanticSubtitle =
    normalizedSemantic.includes('insufficient')
      ? 'Waiting for enough words before judging conversation risk.'
      : semanticStatus;

  const contactLine = phoneNumber
    ? `${resolvedFromContacts ? 'Known contact' : 'Unmatched number'}: ${phoneNumber}`
    : callerId === 'unknown'
      ? 'Caller identity is unknown'
      : 'Caller selected from Trust-Call';

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <View style={styles.callerCard}>
          <Text style={styles.eyebrow}>Protected Call</Text>
          <Text style={styles.callerName}>{callerName}</Text>
          <Text style={styles.contactLine}>{contactLine}</Text>
          <Text style={styles.callTime}>{isCallActive ? formatTime(callDuration) : 'Ready'}</Text>
        </View>

        <View style={[styles.verdictCard, styles[`verdict_${overallTone}`]]}>
          <Text style={styles.verdictLabel}>Trust-Call Verdict</Text>
          <Text style={styles.verdictTitle}>{overallTitle}</Text>
          <Text style={styles.verdictSubtitle}>{overallSubtitle}</Text>
        </View>

        <View style={styles.auditorGrid}>
          <AuditorCard
            title="Voice Authenticity"
            value={signalScore}
            subtitle={
              signalThreat
                ? 'The voice signal looks synthetic or suspicious.'
                : 'Checking whether the audio sounds human.'
            }
            tone={signalThreat ? 'danger' : signalScore.includes('Listening') ? 'neutral' : 'safe'}
          />
          <AuditorCard
            title="Conversation Risk"
            value={semanticStatus}
            subtitle={semanticSubtitle}
            tone={
              normalizedSemantic.includes('coerc') || normalizedSemantic.includes('scam')
                ? 'warning'
                : normalizedSemantic.includes('waiting') || normalizedSemantic.includes('insufficient')
                  ? 'neutral'
                  : 'safe'
            }
          />
          <AuditorCard
            title="Caller Identity"
            value={identityStatus}
            subtitle={identitySubtitle}
            tone={
              normalizedIdentity.includes('mismatch')
                ? 'danger'
                : normalizedIdentity.includes('not enrolled') ||
                    normalizedIdentity.includes('pending') ||
                    normalizedIdentity.includes('checking')
                  ? 'neutral'
                  : 'safe'
            }
          />
        </View>

        <View style={styles.connectionCard}>
          <Text style={styles.connectionTitle}>Connection</Text>
          <Text style={styles.connectionText}>
            {telemetryStatus}
            {isCallActive && audioFrames === 0 ? ' - waiting for microphone audio' : ''}
          </Text>
          {candidateEmbeddingCount > 0 ? (
            <Text style={styles.connectionText}>
              Voice profile candidate: {candidateStatus}
            </Text>
          ) : null}
          <Text style={styles.debugHint}>
            Detailed chunks, frames, buffers, and model metrics are available in Grafana.
          </Text>
        </View>

        <View style={styles.footer}>
          {!isCallActive ? (
            <TouchableOpacity style={styles.acceptButton} onPress={handleAcceptCall}>
              <Text style={styles.actionText}>Accept Protected Call</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={styles.endCallButton} onPress={handleEndCall}>
              <Text style={styles.actionText}>End Call</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </SafeAreaView>
  );
};

const AuditorCard = ({
  title,
  value,
  subtitle,
  tone,
}: {
  title: string;
  value: string;
  subtitle: string;
  tone: VerdictTone;
}) => (
  <View style={styles.auditorCard}>
    <View style={styles.auditorHeader}>
      <Text style={styles.auditorTitle}>{title}</Text>
      <View style={[styles.statusDot, styles[`dot_${tone}`]]} />
    </View>
    <Text style={[styles.auditorValue, styles[`text_${tone}`]]}>{value}</Text>
    <Text style={styles.auditorSubtitle}>{subtitle}</Text>
  </View>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0B0F0E',
  },
  content: {
    flex: 1,
    padding: 16,
    gap: 10,
  },
  callerCard: {
    marginTop: 6,
    paddingVertical: 4,
  },
  eyebrow: {
    color: '#8EA099',
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1.4,
    textTransform: 'uppercase',
  },
  callerName: {
    color: '#F6F4EC',
    fontSize: 32,
    fontWeight: '900',
    marginTop: 6,
  },
  contactLine: {
    color: '#9EA9A4',
    fontSize: 14,
    marginTop: 6,
  },
  callTime: {
    color: '#C8D0CB',
    fontSize: 18,
    fontWeight: '800',
    marginTop: 8,
  },
  verdictCard: {
    borderRadius: 20,
    borderWidth: 1,
    padding: 16,
  },
  verdict_safe: {
    backgroundColor: '#113822',
    borderColor: '#2E9D5B',
  },
  verdict_warning: {
    backgroundColor: '#3D3210',
    borderColor: '#D49A20',
  },
  verdict_danger: {
    backgroundColor: '#461D1D',
    borderColor: '#F05A50',
  },
  verdict_neutral: {
    backgroundColor: '#18211F',
    borderColor: '#2D3A37',
  },
  verdictLabel: {
    color: '#B8C3BE',
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
  verdictTitle: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '900',
    marginTop: 6,
  },
  verdictSubtitle: {
    color: '#DDE5E0',
    fontSize: 13,
    lineHeight: 18,
    marginTop: 6,
  },
  auditorGrid: {
    gap: 8,
  },
  auditorCard: {
    backgroundColor: '#161C1A',
    borderColor: '#2B3531',
    borderRadius: 16,
    borderWidth: 1,
    padding: 13,
  },
  auditorHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  auditorTitle: {
    color: '#BBC7C1',
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },
  auditorValue: {
    fontSize: 18,
    fontWeight: '900',
    marginTop: 7,
  },
  auditorSubtitle: {
    color: '#A5B1AB',
    fontSize: 12,
    lineHeight: 16,
    marginTop: 4,
  },
  statusDot: {
    borderRadius: 6,
    height: 12,
    width: 12,
  },
  dot_safe: {
    backgroundColor: '#55C878',
  },
  dot_warning: {
    backgroundColor: '#F1B63B',
  },
  dot_danger: {
    backgroundColor: '#FF5B52',
  },
  dot_neutral: {
    backgroundColor: '#66736E',
  },
  text_safe: {
    color: '#55C878',
  },
  text_warning: {
    color: '#F1B63B',
  },
  text_danger: {
    color: '#FF5B52',
  },
  text_neutral: {
    color: '#DCE3DF',
  },
  connectionCard: {
    backgroundColor: '#111615',
    borderColor: '#26302D',
    borderRadius: 14,
    borderWidth: 1,
    padding: 12,
  },
  connectionTitle: {
    color: '#E6ECE8',
    fontSize: 15,
    fontWeight: '800',
  },
  connectionText: {
    color: '#AEB8B3',
    fontSize: 14,
    marginTop: 6,
  },
  debugHint: {
    color: '#6F7C76',
    fontSize: 11,
    lineHeight: 15,
    marginTop: 6,
  },
  footer: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  acceptButton: {
    alignItems: 'center',
    backgroundColor: '#2E9D5B',
    borderRadius: 16,
    paddingVertical: 15,
  },
  endCallButton: {
    alignItems: 'center',
    backgroundColor: '#D83B35',
    borderRadius: 16,
    paddingVertical: 15,
  },
  actionText: {
    color: '#FFFFFF',
    fontSize: 17,
    fontWeight: '900',
  },
});

export default CallScreen;
