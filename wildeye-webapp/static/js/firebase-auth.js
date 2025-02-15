import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.8.0/firebase-app.js';
import {
    getAuth,
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    signOut,
    onAuthStateChanged,
    sendPasswordResetEmail,
    GoogleAuthProvider,
    signInWithPopup
} from 'https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js';

const firebaseConfig = {
    apiKey: "AIzaSyCQJKf9jhxlmGXLVTV_DYlFaVxbzj8EjDk",
    authDomain: "wildeye-finalyear.firebaseapp.com",
    projectId: "wildeye-finalyear",
    storageBucket: "wildeye-finalyear.firebasestorage.app",
    messagingSenderId: "836572506957",
    appId: "1:836572506957:web:22eba4a18f00868032e504",
    measurementId: "G-X4WDV45CXR"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// Auth state observer
export function initAuthStateObserver() {
    const publicPages = ['/', '/about', '/contact', '/faq'];
    const currentPath = window.location.pathname;
    
    onAuthStateChanged(auth, (user) => {
        if (user) {
            if (currentPath === '/') {
                window.location.href = '/home';
            }
        } else {
            if (!publicPages.includes(currentPath)) {
                window.location.href = '/';
            }
        }
    });
}

// Sign in with email/password
export async function loginWithEmailAndPassword(email, password) {
    try {
        await signInWithEmailAndPassword(auth, email, password);
        window.location.href = '/home';
    } catch (error) {
        console.error('Login error:', error);
        throw new Error('Invalid email or password.');
    }
}

// Register with email/password
export async function registerWithEmailAndPassword(email, password) {
    try {
        await createUserWithEmailAndPassword(auth, email, password);
        return true;
    } catch (error) {
        console.error('Registration error:', error);
        throw new Error(error.message);
    }
}

// Google sign in
export async function signInWithGoogle() {
    try {
        provider.setCustomParameters({
            prompt: 'select_account'
        });
        await signInWithPopup(auth, provider);
        window.location.href = '/home';
    } catch (error) {
        console.error('Google sign-in error:', error);
        throw error;
    }
}

// Password reset
export async function resetPassword(email) {
    try {
        await sendPasswordResetEmail(auth, email);
        return true;
    } catch (error) {
        console.error('Password reset error:', error);
        throw new Error('Failed to send password reset email: ' + error.message);
    }
}

// Handle sign out
export async function logout() {
    try {
        await signOut(auth);
        window.location.href = '/';
    } catch (error) {
        console.error('Logout error:', error);
        throw new Error('Failed to log out. Please try again.');
    }
}