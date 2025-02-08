import { initializeApp } from 'https://www.gstatic.com/firebasejs/9.6.1/firebase-app.js';
import { 
    getAuth, 
    signInWithEmailAndPassword, 
    createUserWithEmailAndPassword, 
    signOut, 
    onAuthStateChanged,
    sendPasswordResetEmail,
    GoogleAuthProvider,
    signInWithPopup
} from 'https://www.gstatic.com/firebasejs/9.6.1/firebase-auth.js';

// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyCQJKf9jhxlmGXLVTV_DYlFaVxbzj8EjDk",
    authDomain: "wildeye-finalyear.firebaseapp.com",
    projectId: "wildeye-finalyear",
    storageBucket: "wildeye-finalyear.firebasestorage.app",
    messagingSenderId: "836572506957",
    appId: "1:836572506957:web:22eba4a18f00868032e504",
    measurementId: "G-X4WDV45CXR"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// Auth state observer
export function initAuthStateObserver() {
    onAuthStateChanged(auth, (user) => {
        if (!user && window.location.pathname !== '/') {
            window.location.href = '/';
        }
    });
}

// Login function
export async function loginWithEmailAndPassword(email, password) {
    try {
        await signInWithEmailAndPassword(auth, email, password);
        window.location.href = '/home';
    } catch (error) {
        throw new Error('Invalid email or password.');
    }
}

// Register function
export async function registerWithEmailAndPassword(email, password) {
    try {
        await createUserWithEmailAndPassword(auth, email, password);
        return true;
    } catch (error) {
        throw new Error('Error creating account: ' + error.message);
    }
}

// Google sign in
export async function signInWithGoogle() {
    try {
        await signInWithPopup(auth, provider);
        window.location.href = '/home';
    } catch (error) {
        throw new Error('Google sign-in failed: ' + error.message);
    }
}

// Password reset
export async function resetPassword(email) {
    try {
        await sendPasswordResetEmail(auth, email);
        return true;
    } catch (error) {
        throw new Error('Error: ' + error.message);
    }
}

// Logout function
export async function logout() {
    try {
        await signOut(auth);
        window.location.href = '/';
    } catch (error) {
        throw new Error('Error logging out. Please try again.');
    }
}