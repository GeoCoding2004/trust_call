import { NativeModules, Platform } from 'react-native';

const BACKEND_PORT = 8080;
const ANDROID_EMULATOR_HOST = '10.0.2.2';
const IOS_SIMULATOR_HOST = '127.0.0.1';
const CLOUD_BACKEND_BASE_URL: string | null =
  'http://35.189.221.158:8080';

// Set this to a LAN IP like "192.168.1.10" when you want to force a physical-device target.
const MANUAL_BACKEND_HOST_OVERRIDE: string | null = '127.0.0.1';

function normalizePath(path: string): string {
  return path.startsWith('/') ? path : `/${path}`;
}

function extractMetroHost(): string | null {
  const scriptURL =
    NativeModules.SourceCode?.scriptURL ??
    NativeModules.SourceCode?.getConstants?.().scriptURL ??
    null;

  if (!scriptURL || typeof scriptURL !== 'string') {
    return null;
  }

  try {
    const parsed = new URL(scriptURL);
    return parsed.hostname || null;
  } catch {
    const match = scriptURL.match(/^https?:\/\/([^/:]+)/i);
    return match?.[1] ?? null;
  }
}

function resolveBackendHost(): string {
  if (MANUAL_BACKEND_HOST_OVERRIDE) {
    return MANUAL_BACKEND_HOST_OVERRIDE;
  }

  const metroHost = extractMetroHost();
  if (metroHost && !['localhost', '127.0.0.1'].includes(metroHost)) {
    return metroHost;
  }

  if (Platform.OS === 'android') {
    return ANDROID_EMULATOR_HOST;
  }

  return IOS_SIMULATOR_HOST;
}

export function getResolvedBackendBaseUrl(): string {
  if (CLOUD_BACKEND_BASE_URL) {
    return CLOUD_BACKEND_BASE_URL.replace(/\/+$/, '');
  }

  return `http://${resolveBackendHost()}:${BACKEND_PORT}`;
}

export function getResolvedBackendWsBaseUrl(): string {
  if (CLOUD_BACKEND_BASE_URL) {
    return CLOUD_BACKEND_BASE_URL.replace(/^http/i, 'ws').replace(/\/+$/, '');
  }

  return `ws://${resolveBackendHost()}:${BACKEND_PORT}`;
}

export function getBackendHttpUrl(path: string): string {
  return `${getResolvedBackendBaseUrl()}${normalizePath(path)}`;
}

export function getBackendWsUrl(path: string): string {
  return `${getResolvedBackendWsBaseUrl()}${normalizePath(path)}`;
}
