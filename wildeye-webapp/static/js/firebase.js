// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyCIDC0y-ggQxTuvM9FX2u_Q7LCE2QGnSU4",
    authDomain: "wildeye007-58f76.firebaseapp.com",
    projectId: "wildeye007-58f76",
    storageBucket: "wildeye007-58f76.firebasestorage.app",
    messagingSenderId: "539234224023",
    appId: "1:539234224023:web:3361ab33852cbf097166d2",
    measurementId: "G-4Z19900X33"
  };  

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Signup function
function signUp() {
    const email = document.getElementById("signup-email").value;
    const password = document.getElementById("signup-password").value;

    firebase.auth().createUserWithEmailAndPassword(email, password)
        .then((userCredential) => {
            alert("Signup successful!");
        })
        .catch((error) => {
            alert(error.message);
        });
}

// Login function
function logIn() {
    const email = document.getElementById("login-email").value;
    const password = document.getElementById("login-password").value;

    firebase.auth().signInWithEmailAndPassword(email, password)
        .then((userCredential) => {
            window.location.href = "/home"; // Redirect to home
        })
        .catch((error) => {
            alert(error.message);
        });
}

// Logout function
function logOut() {
    firebase.auth().signOut()
        .then(() => {
            alert("Logged out successfully!");
            window.location.href = "/";
        })
        .catch((error) => {
            alert(error.message);
        });
}
