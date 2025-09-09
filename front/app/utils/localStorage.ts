import type { TranslationSettings, FinishedImage } from '@/types';

const SETTINGS_KEY = 'manga-translator-settings';
const FINISHED_IMAGES_KEY = 'manga-translator-finished-images';

export const loadSettings = (): Partial<TranslationSettings> => {
  try {
    const stored = localStorage.getItem(SETTINGS_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch (error) {
    console.warn('Failed to load settings from localStorage:', error);
    return {};
  }
};

export const saveSettings = (settings: TranslationSettings): void => {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (error) {
    console.warn('Failed to save settings to localStorage:', error);
  }
};

export const loadFinishedImages = (): FinishedImage[] => {
  try {
    const stored = localStorage.getItem(FINISHED_IMAGES_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (error) {
    console.warn('Failed to load finished images from localStorage:', error);
    return [];
  }
};

export const saveFinishedImages = (images: FinishedImage[]): void => {
  try {
    // Keep only the last 50 images to prevent localStorage from getting too large
    const limitedImages = images.slice(-50);
    localStorage.setItem(FINISHED_IMAGES_KEY, JSON.stringify(limitedImages));
  } catch (error) {
    console.warn('Failed to save finished images to localStorage:', error);
  }
};

export const addFinishedImage = (image: FinishedImage): void => {
  try {
    const existing = loadFinishedImages();
    const updated = [image, ...existing]; // Add new image at the top
    saveFinishedImages(updated);
  } catch (error) {
    console.warn('Failed to add finished image to localStorage:', error);
  }
}; 