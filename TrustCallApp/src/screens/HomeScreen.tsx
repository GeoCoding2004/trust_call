import React, { useEffect, useState } from 'react';
import {
  Alert,
  NativeModules,
  PermissionsAndroid,
  Platform,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { getBackendHttpUrl } from '../config/backend';

type PhoneContact = {
  id: string;
  name: string;
  phoneNumber?: string;
};

type TrustedContact = {
  callerId: string;
  callerName: string;
  label: string;
  phoneNumber?: string;
  phoneDigits?: string;
  source: 'phone' | 'demo';
};

type EnrollmentStatus = {
  caller_id: string;
  enrolled: boolean;
  embedding_dim?: number;
  updated_at_utc?: string | null;
  num_updates?: number;
};

type TrustCallContactsModule = {
  getContacts: () => Promise<PhoneContact[]>;
};

const { TrustCallContacts } = NativeModules as {
  TrustCallContacts?: TrustCallContactsModule;
};

const DEMO_CONTACTS: TrustedContact[] = [
  {
    callerId: 'alice_demo',
    callerName: 'Alice Demo',
    label: '+1 555 0101',
    phoneNumber: '+1 555 0101',
    phoneDigits: '15550101',
    source: 'demo',
  },
  {
    callerId: 'mom_demo',
    callerName: 'Mom Demo',
    label: '+1 555 0102',
    phoneNumber: '+1 555 0102',
    phoneDigits: '15550102',
    source: 'demo',
  },
  {
    callerId: 'bank_contact',
    callerName: 'Bank Contact',
    label: '+1 555 0103',
    phoneNumber: '+1 555 0103',
    phoneDigits: '15550103',
    source: 'demo',
  },
];

const UNKNOWN_INCOMING_CALL = {
  callerId: 'unknown',
  callerName: 'Unknown Caller',
};

const normalizePhoneDigits = (value?: string): string => (value ?? '').replace(/\D/g, '');

const callerIdFromPhoneDigits = (digits: string): string => `contact_${digits || 'unknown'}`;

const normalizeCallerId = (contact: PhoneContact): string => {
  const phoneDigits = normalizePhoneDigits(contact.phoneNumber);
  const stableValue = phoneDigits || contact.id || contact.name;
  const normalized = stableValue.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  return `contact_${normalized || 'unknown'}`;
};

const toTrustedContact = (contact: PhoneContact): TrustedContact => {
  const phoneDigits = normalizePhoneDigits(contact.phoneNumber);
  return {
    callerId: phoneDigits ? callerIdFromPhoneDigits(phoneDigits) : normalizeCallerId(contact),
    callerName: contact.name || contact.phoneNumber || 'Unnamed Contact',
    label: contact.phoneNumber || 'Phone contact',
    phoneNumber: contact.phoneNumber,
    phoneDigits,
    source: 'phone',
  };
};

const phoneNumbersMatch = (left?: string, right?: string) => {
  const leftDigits = normalizePhoneDigits(left);
  const rightDigits = normalizePhoneDigits(right);
  if (!leftDigits || !rightDigits) return false;
  if (leftDigits === rightDigits) return true;

  const suffixLength = Math.min(leftDigits.length, rightDigits.length, 10);
  return suffixLength >= 7 && leftDigits.slice(-suffixLength) === rightDigits.slice(-suffixLength);
};

const requestAndroidPermission = async (
  permission: string,
  title: string,
  message: string,
) => {
  if (Platform.OS !== 'android') return true;

  const granted = await PermissionsAndroid.request(permission as any, {
    title,
    message,
    buttonNeutral: 'Ask Me Later',
    buttonNegative: 'Cancel',
    buttonPositive: 'OK',
  });

  return granted === PermissionsAndroid.RESULTS.GRANTED;
};

const requestCallPermissions = async () => {
  try {
    const granted = await requestAndroidPermission(
      PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
      'Trust Call Microphone Permission',
      'Trust Call needs microphone access to analyze live call audio.',
    );

    if (!granted) {
      Alert.alert(
        'Microphone Required',
        'Trust Call needs microphone access before it can verify the caller voice.',
      );
    }
    return granted;
  } catch (err) {
    console.warn('Error requesting microphone permission:', err);
    return false;
  }
};

const requestContactsPermission = async () => {
  try {
    const granted = await requestAndroidPermission(
      PermissionsAndroid.PERMISSIONS.READ_CONTACTS,
      'Trust Call Contacts Permission',
      'Trust Call uses contacts to match incoming numbers to trusted callers.',
    );

    if (!granted) {
      Alert.alert(
        'Contacts Unavailable',
        'Trust Call will show demo contacts until contacts access is allowed.',
      );
    }
    return granted;
  } catch (err) {
    console.warn('Error requesting contacts permission:', err);
    return false;
  }
};

const formatStatusDetail = (status?: EnrollmentStatus) => {
  if (!status) return 'Checking protection';
  if (!status.enrolled) return 'Voice profile not saved yet';

  const updates = status.num_updates ?? 0;
  return updates > 0 ? 'Protected voice profile ready' : 'Voice profile ready';
};

const HomeScreen = ({ navigation }: any) => {
  const [contacts, setContacts] = useState<TrustedContact[]>(DEMO_CONTACTS);
  const [selectedCallerId, setSelectedCallerId] = useState(DEMO_CONTACTS[0].callerId);
  const [enrollmentByCallerId, setEnrollmentByCallerId] = useState<
    Record<string, EnrollmentStatus | undefined>
  >({});
  const [backendStatus, setBackendStatus] = useState('Checking backend');
  const [, setContactsStatus] = useState('Demo contacts loaded');
  const [searchText, setSearchText] = useState('');
  const [incomingPhoneNumber, setIncomingPhoneNumber] = useState('+1 555 0101');
  const [contactsExpanded, setContactsExpanded] = useState(false);

  const selectedContact =
    contacts.find((contact) => contact.callerId === selectedCallerId) ?? contacts[0];

  const visibleContactsLimit = contactsExpanded || searchText.trim() ? 50 : 3;
  const visibleContacts = contacts.filter((contact) => {
    const query = searchText.trim().toLowerCase();
    if (!query) return true;
    return (
      contact.callerName.toLowerCase().includes(query) ||
      contact.label.toLowerCase().includes(query)
    );
  }).slice(0, visibleContactsLimit);

  const resolveIncomingContact = (phoneNumber: string): TrustedContact | undefined =>
    contacts.find((contact) => phoneNumbersMatch(contact.phoneNumber ?? contact.label, phoneNumber));

  const refreshEnrollmentStatuses = async (contactsToCheck = contacts) => {
    if (contactsToCheck.length === 0) return;

    setBackendStatus('Checking backend');
    try {
      const entries = await Promise.all(
        contactsToCheck.map(async (contact) => {
          const response = await fetch(
            getBackendHttpUrl(`/identity/enrollment/${contact.callerId}`),
          );
          if (!response.ok) {
            throw new Error(`Enrollment lookup failed for ${contact.callerId}`);
          }
          const status = (await response.json()) as EnrollmentStatus;
          return [contact.callerId, status] as const;
        }),
      );

      setEnrollmentByCallerId(Object.fromEntries(entries));
      setBackendStatus('AI auditors online');
    } catch (error) {
      console.warn('Failed to refresh enrollment statuses:', error);
      setBackendStatus('Backend offline');
    }
  };

  const loadPhoneContacts = async () => {
    if (Platform.OS !== 'android' || !TrustCallContacts) {
      setContacts(DEMO_CONTACTS);
      setSelectedCallerId(DEMO_CONTACTS[0].callerId);
      setContactsStatus('Demo contacts loaded');
      await refreshEnrollmentStatuses(DEMO_CONTACTS);
      return;
    }

    const hasContactsPermission = await requestContactsPermission();
    if (!hasContactsPermission) {
      setContacts(DEMO_CONTACTS);
      setSelectedCallerId(DEMO_CONTACTS[0].callerId);
      setContactsStatus('Demo contacts loaded');
      await refreshEnrollmentStatuses(DEMO_CONTACTS);
      return;
    }

    try {
      const phoneContacts = await TrustCallContacts.getContacts();
      const trustedContacts = phoneContacts.map(toTrustedContact).slice(0, 100);

      if (trustedContacts.length === 0) {
        setContacts(DEMO_CONTACTS);
        setSelectedCallerId(DEMO_CONTACTS[0].callerId);
        setContactsStatus('No phone contacts found');
        await refreshEnrollmentStatuses(DEMO_CONTACTS);
        return;
      }

      setContacts(trustedContacts);
      setSelectedCallerId(trustedContacts[0].callerId);
      setContactsStatus(`${trustedContacts.length} phone contacts loaded`);
      await refreshEnrollmentStatuses(trustedContacts);
    } catch (error) {
      console.warn('Failed to load phone contacts:', error);
      setContacts(DEMO_CONTACTS);
      setSelectedCallerId(DEMO_CONTACTS[0].callerId);
      setContactsStatus('Demo contacts loaded');
      await refreshEnrollmentStatuses(DEMO_CONTACTS);
    }
  };

  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', loadPhoneContacts);
    loadPhoneContacts();
    return unsubscribe;
    // loadPhoneContacts is intentionally re-created from current screen state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [navigation]);

  const startSelectedContactCall = async () => {
    const hasPermission = await requestCallPermissions();
    if (!hasPermission || !selectedContact) return;

    navigation.navigate('CallScreen', {
      callerId: selectedContact.callerId,
      callerName: selectedContact.callerName,
      phoneNumber: selectedContact.phoneNumber,
      resolvedFromContacts: selectedContact.source === 'phone',
    });
  };

  const startIncomingPhoneCall = async () => {
    const digits = normalizePhoneDigits(incomingPhoneNumber);
    if (!digits) {
      Alert.alert('Incoming Number Required', 'Enter the caller phone number to resolve it.');
      return;
    }

    const hasPermission = await requestCallPermissions();
    if (!hasPermission) return;

    const resolvedContact = resolveIncomingContact(incomingPhoneNumber);
    const caller = resolvedContact ?? {
      callerId: callerIdFromPhoneDigits(digits),
      callerName: `Unknown ${incomingPhoneNumber.trim()}`,
      label: incomingPhoneNumber.trim(),
      phoneNumber: incomingPhoneNumber.trim(),
      phoneDigits: digits,
      source: 'demo' as const,
    };

    navigation.navigate('CallScreen', {
      callerId: caller.callerId,
      callerName: caller.callerName,
      phoneNumber: caller.phoneNumber,
      resolvedFromContacts: Boolean(resolvedContact),
    });
  };

  const startUnknownCallerCall = async () => {
    const hasPermission = await requestCallPermissions();
    if (!hasPermission) return;

    navigation.navigate('CallScreen', UNKNOWN_INCOMING_CALL);
  };

  const selectedStatus = selectedContact
    ? enrollmentByCallerId[selectedContact.callerId]
    : undefined;

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.hero}>
          <Text style={styles.appName}>Trust-Call</Text>
          <Text style={styles.heroSubtitle}>Live AI protection for suspicious calls</Text>
        </View>

        <View style={styles.statusCard}>
          <View>
            <Text style={styles.statusTitle}>Protection Status</Text>
            <Text
              style={[
                styles.statusActive,
                backendStatus === 'Backend offline' && styles.statusOffline,
              ]}>
              {backendStatus}
            </Text>
          </View>
          <View style={styles.statusDot} />
        </View>

        <View style={styles.incomingPanel}>
          <Text style={styles.selectedEyebrow}>Check a Call</Text>
          <TextInput
            style={styles.numberInput}
            placeholder="Incoming phone number"
            placeholderTextColor="#777"
            keyboardType="phone-pad"
            value={incomingPhoneNumber}
            onChangeText={setIncomingPhoneNumber}
          />
          <TouchableOpacity style={styles.primaryButton} onPress={startIncomingPhoneCall}>
            <Text style={styles.primaryButtonText}>Check Incoming Call</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.quickActions}>
          <TouchableOpacity style={styles.quickAction} onPress={startSelectedContactCall}>
            <Text style={styles.quickActionLabel}>Selected Contact</Text>
            <Text style={styles.quickActionTitle}>{selectedContact?.callerName ?? 'Contact'}</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.quickAction} onPress={startUnknownCallerCall}>
            <Text style={styles.quickActionLabel}>Demo Test</Text>
            <Text style={styles.quickActionTitle}>Unknown Caller</Text>
          </TouchableOpacity>
        </View>

        {selectedContact ? (
          <View style={styles.selectedPanel}>
            <Text style={styles.selectedEyebrow}>Selected Contact</Text>
            <Text style={styles.selectedName}>{selectedContact.callerName}</Text>
            <Text style={styles.selectedStatus}>{formatStatusDetail(selectedStatus)}</Text>
          </View>
        ) : null}

        <View style={styles.sectionHeader}>
          <TouchableOpacity
            style={styles.contactsToggle}
            onPress={() => setContactsExpanded((current) => !current)}>
            <View>
              <Text style={styles.sectionTitle}>Contacts</Text>
              <Text style={styles.sectionSubtitle}>
                {contactsExpanded ? 'Tap to collapse' : 'Tap to choose another caller'}
              </Text>
            </View>
            <Text style={styles.expandIcon}>{contactsExpanded ? '−' : '+'}</Text>
          </TouchableOpacity>
        </View>

        {contactsExpanded ? (
          <>
            <View style={styles.contactsToolbar}>
              <TextInput
                style={[styles.searchInput, styles.contactsSearchInput]}
                placeholder="Search contacts"
                placeholderTextColor="#777"
                value={searchText}
                onChangeText={setSearchText}
              />
              <TouchableOpacity style={styles.refreshButton} onPress={loadPhoneContacts}>
                <Text style={styles.refreshButtonText}>Refresh</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.contactList}>
              {visibleContacts.map((contact, index) => {
                const status = enrollmentByCallerId[contact.callerId];
                const isSelected = contact.callerId === selectedContact?.callerId;
                const isEnrolled = status?.enrolled === true;

                return (
                  <TouchableOpacity
                    key={`${contact.callerId}-${contact.label}-${index}`}
                    style={[styles.contactCard, isSelected && styles.contactCardSelected]}
                    onPress={() => {
                      setSelectedCallerId(contact.callerId);
                      setContactsExpanded(false);
                      setSearchText('');
                    }}>
                    <View style={styles.contactTextBlock}>
                      <Text style={styles.contactName}>{contact.callerName}</Text>
                      <Text style={styles.contactLabel}>{contact.label}</Text>
                    </View>
                    <View
                      style={[
                        styles.enrollmentPill,
                        isEnrolled ? styles.enrollmentPillReady : styles.enrollmentPillMissing,
                      ]}>
                      <Text style={styles.enrollmentPillText}>
                        {isEnrolled ? 'Protected' : 'New'}
                      </Text>
                    </View>
                  </TouchableOpacity>
                );
              })}
            </View>
            {contacts.length > visibleContacts.length ? (
              <Text style={styles.contactHint}>
                Showing {visibleContacts.length} of {contacts.length}. Search to find another contact.
              </Text>
            ) : null}
          </>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#090D0C',
  },
  content: {
    padding: 16,
    paddingBottom: 28,
  },
  hero: {
    marginTop: 10,
    marginBottom: 12,
  },
  appName: {
    color: '#F7F5EE',
    fontSize: 34,
    fontWeight: '900',
    letterSpacing: -1,
  },
  heroSubtitle: {
    color: '#9AA7A1',
    fontSize: 14,
    fontWeight: '600',
    marginTop: 4,
  },
  statusCard: {
    alignItems: 'center',
    backgroundColor: '#151B19',
    borderWidth: 1,
    borderColor: '#26322E',
    borderRadius: 18,
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 16,
  },
  statusTitle: {
    color: '#89968F',
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  statusActive: {
    color: '#58C879',
    fontSize: 18,
    fontWeight: '900',
    marginTop: 4,
    textTransform: 'capitalize',
  },
  statusOffline: {
    color: '#FF3B30',
  },
  statusDot: {
    backgroundColor: '#58C879',
    borderRadius: 8,
    height: 16,
    width: 16,
  },
  incomingPanel: {
    backgroundColor: '#102119',
    borderColor: '#2E9D5B',
    borderRadius: 22,
    borderWidth: 1,
    gap: 10,
    marginTop: 14,
    padding: 16,
  },
  numberInput: {
    backgroundColor: '#151716',
    borderColor: '#303B37',
    borderRadius: 14,
    borderWidth: 1,
    color: '#FFFFFF',
    fontSize: 22,
    fontWeight: '900',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  quickActions: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 12,
  },
  quickAction: {
    backgroundColor: '#151B19',
    borderColor: '#26322E',
    borderRadius: 18,
    borderWidth: 1,
    flex: 1,
    padding: 14,
  },
  quickActionLabel: {
    color: '#7F8D86',
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 0.9,
    textTransform: 'uppercase',
  },
  quickActionTitle: {
    color: '#F1F5F2',
    fontSize: 15,
    fontWeight: '900',
    marginTop: 6,
  },
  sectionHeader: {
    marginBottom: 8,
    marginTop: 18,
  },
  contactsToggle: {
    alignItems: 'center',
    backgroundColor: '#151B19',
    borderColor: '#26322E',
    borderRadius: 18,
    borderWidth: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 14,
  },
  sectionTitle: {
    color: '#F5F5F5',
    fontSize: 20,
    fontWeight: '900',
  },
  sectionSubtitle: {
    color: '#7F8D86',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
  },
  expandIcon: {
    color: '#F5F5F5',
    fontSize: 28,
    fontWeight: '700',
  },
  contactsToolbar: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: 10,
    marginBottom: 10,
  },
  refreshButton: {
    borderWidth: 1,
    borderColor: '#2B3531',
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  refreshButtonText: {
    color: '#CFCFCF',
    fontSize: 12,
    fontWeight: '800',
  },
  searchInput: {
    backgroundColor: '#151716',
    borderColor: '#303B37',
    borderRadius: 14,
    borderWidth: 1,
    color: '#FFFFFF',
    fontSize: 15,
    marginBottom: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  contactsSearchInput: {
    flex: 1,
    marginBottom: 0,
  },
  contactList: {
    gap: 8,
  },
  contactCard: {
    alignItems: 'center',
    backgroundColor: '#151716',
    borderColor: '#2B3531',
    borderRadius: 16,
    borderWidth: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 13,
  },
  contactCardSelected: {
    backgroundColor: '#112018',
    borderColor: '#45B765',
  },
  contactTextBlock: {
    flex: 1,
    paddingRight: 12,
  },
  contactName: {
    color: '#F2F2F2',
    fontSize: 17,
    fontWeight: '900',
  },
  contactLabel: {
    color: '#9F9F9F',
    fontSize: 12,
    marginTop: 3,
  },
  enrollmentPill: {
    borderRadius: 999,
    paddingHorizontal: 9,
    paddingVertical: 5,
  },
  enrollmentPillReady: {
    backgroundColor: '#2E7D32',
  },
  enrollmentPillMissing: {
    backgroundColor: '#4A4A4A',
  },
  enrollmentPillText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  selectedPanel: {
    backgroundColor: '#151B19',
    borderColor: '#26322E',
    borderRadius: 18,
    borderWidth: 1,
    marginTop: 12,
    padding: 14,
  },
  selectedEyebrow: {
    color: '#88958F',
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  selectedName: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '900',
    marginTop: 6,
  },
  selectedStatus: {
    color: '#CFCFCF',
    fontSize: 13,
    marginTop: 4,
  },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: '#45B765',
    borderRadius: 15,
    paddingVertical: 14,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '900',
  },
  contactHint: {
    color: '#7B8781',
    fontSize: 12,
    marginTop: 10,
    textAlign: 'center',
  },
});

export default HomeScreen;
