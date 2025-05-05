// Map variables in global scope for easier access
window.map = null;
window.marker = null;
window.radiusCircle = null;

// Track changes that need to be saved
let pendingChanges = {};
let isSaving = false;

// Function to create Google Maps link from coordinates
function createGoogleMapsLink(lat, lng) {
    return `https://www.google.com/maps?q=${lat},${lng}`;
}

// Debounce function to prevent excessive API calls
function debounce(func, wait) {
    let timeout;
    return function() {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}

// Function to show saving indicator
function showSaving() {
    const saveIndicator = document.getElementById('saveIndicator');
    const savingSpinner = document.getElementById('savingSpinner');
    const saveText = document.getElementById('saveText');
    
    if (saveIndicator && savingSpinner && saveText) {
        saveIndicator.classList.add('visible');
        savingSpinner.classList.add('spinner');
        saveText.textContent = 'Saving...';
    }
    isSaving = true;
}

// Function to show saved indicator
function showSaved() {
    const saveIndicator = document.getElementById('saveIndicator');
    const savingSpinner = document.getElementById('savingSpinner');
    const saveText = document.getElementById('saveText');
    
    if (saveIndicator && savingSpinner && saveText) {
        savingSpinner.classList.remove('spinner');
        saveText.textContent = 'Saved';
        
        // Hide indicator after a delay
        setTimeout(() => {
            saveIndicator.classList.remove('visible');
        }, 2000);
    }
    isSaving = false;
    pendingChanges = {}; // Clear pending changes after successful save
}

// Function to show error
function showError(error) {
    const saveIndicator = document.getElementById('saveIndicator');
    const savingSpinner = document.getElementById('savingSpinner');
    const saveText = document.getElementById('saveText');
    
    if (saveIndicator && savingSpinner && saveText) {
        savingSpinner.classList.remove('spinner');
        saveText.textContent = 'Error saving';
    }
    
    // Show error notification
    Swal.fire({
        title: 'Error Saving Settings',
        text: error,
        icon: 'error',
        confirmButtonText: 'OK'
    });
    
    // Hide indicator after a delay
    if (saveIndicator) {
        setTimeout(() => {
            saveIndicator.classList.remove('visible');
        }, 3000);
    }
    isSaving = false;
}

// Helper function to convert dial code to country code
function getCountryCodeByDialCode(dialCode) {
    if (window.intlTelInputGlobals && window.intlTelInputGlobals.getCountryData) {
        var countryData = window.intlTelInputGlobals.getCountryData();
        for (var i = 0; i < countryData.length; i++) {
            if (countryData[i].dialCode == dialCode) {
                return countryData[i].iso2;
            }
        }
    }
    return "in"; // Default to India if not found
}

// Function to extract coordinates from a Google Maps link
function extractCoordinatesFromMapsLink(mapsLink) {
    if (!mapsLink || typeof mapsLink !== 'string') {
        return null;
    }
    
    // Accept direct coordinates input (lat,lng format)
    if (mapsLink.includes(',') && !mapsLink.includes('http')) {
        try {
            const parts = mapsLink.split(',');
            if (parts.length === 2) {
                const lat = parseFloat(parts[0].trim());
                const lng = parseFloat(parts[1].trim());
                
                if (!isNaN(lat) && !isNaN(lng) && 
                    lat >= -90 && lat <= 90 && 
                    lng >= -180 && lng <= 180) {
                    return {
                        lat: lat,
                        lng: lng
                    };
                }
            }
        } catch (e) {
            console.error("Error parsing coordinates:", e);
        }
    }
    
    // Accept both standard and mobile Google Maps links
    if (mapsLink.includes('google.com/maps') || 
        mapsLink.includes('maps.app.goo.gl') || 
        mapsLink.includes('goo.gl/maps')) {
        
        // Common patterns in Google Maps URLs
        const patterns = [
            /[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)/,  // ?q=lat,lng
            /@(-?\d+\.\d+),(-?\d+\.\d+)/,        // @lat,lng
            /ll=(-?\d+\.\d+),(-?\d+\.\d+)/,      // ll=lat,lng
            /data=.*!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)/ // data=...!3d{lat}!4d{lng}
        ];
        
        for (const pattern of patterns) {
            const match = mapsLink.match(pattern);
            if (match && match.length >= 3) {
                const lat = parseFloat(match[1]);
                const lng = parseFloat(match[2]);
                
                if (!isNaN(lat) && !isNaN(lng) && 
                    lat >= -90 && lat <= 90 && 
                    lng >= -180 && lng <= 180) {
                    return {
                        lat: lat,
                        lng: lng
                    };
                }
            }
        }
    }
    
    // If we couldn't extract coordinates but it looks like a valid maps link, 
    // accept it anyway and let the backend handle it
    if (mapsLink.includes('maps.app.goo.gl') || 
        mapsLink.includes('goo.gl/maps') || 
        mapsLink.includes('google.com/maps')) {
        console.log("Accepting Google Maps link without extracting coordinates:", mapsLink);
        return true; // Return true to indicate it's a valid link even though we couldn't extract coordinates
    }
    
    return null;
}

// Calculate distance between two coordinates using Haversine formula
function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Radius of the earth in km
    const dLat = deg2rad(lat2 - lat1);
    const dLon = deg2rad(lon2 - lon1); 
    const a = 
        Math.sin(dLat/2) * Math.sin(dLat/2) +
        Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) * 
        Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)); 
    const distance = R * c; // Distance in km
    return distance;
}

function deg2rad(deg) {
    return deg * (Math.PI/180);
}

// Function to get current animal selections
function getSelectedAnimals() {
    const animalAllCheckbox = document.getElementById('animal_all');
    if (!animalAllCheckbox) return ['all'];
    
    const animalCheckboxes = document.querySelectorAll('input[id^="animal_"]:not([id="animal_all"])');
    
    if (animalAllCheckbox.checked) {
        return ['all'];
    }
    
    const selected = [];
    animalCheckboxes.forEach(checkbox => {
        if (checkbox.checked) {
            selected.push(checkbox.getAttribute('data-animal'));
        }
    });
    
    return selected.length > 0 ? selected : ['all']; // Default to 'all' if none selected
}

// Function to save animal selections
function saveAnimalSelections() {
    const selected = getSelectedAnimals();
    console.log("Saving animal selections:", selected);
    saveSettings({ 'broadcast.animals': selected });
}

// Function to get notification methods
function getSelectedMethods() {
    const methods = [];
    document.querySelectorAll('.auto-save-method').forEach(checkbox => {
        if (checkbox.checked) {
            methods.push(checkbox.getAttribute('data-method'));
        }
    });
    
    return methods.length > 0 ? methods : ['email', 'sms']; // Default to email and SMS if none selected
}

// Function to save notification methods
function saveNotificationMethods() {
    const methods = getSelectedMethods();
    console.log("Saving notification methods:", methods);
    saveSettings({ 'broadcast.methods': methods });
}

// Function to update individual checkboxes based on "All animals" state
function updateAnimalCheckboxes() {
    const animalAllCheckbox = document.getElementById('animal_all');
    if (!animalAllCheckbox) return;
    
    const animalCheckboxes = document.querySelectorAll('input[id^="animal_"]:not([id="animal_all"])');
    
    const isAllChecked = animalAllCheckbox.checked;
    console.log("All animals checkbox changed:", isAllChecked);
    
    animalCheckboxes.forEach(checkbox => {
        checkbox.checked = isAllChecked;
        checkbox.disabled = isAllChecked;  // Disable individual when "All" is checked
    });
    
    // Save animal selections after updates
    saveAnimalSelections();
}

// Function for initial setup when page loads
function setupLocationInput() {
    const locationInput = document.getElementById('broadcast_location');
    if (!locationInput) return;
    
    console.log("Setting up location input with value:", locationInput.value);
    
    // If we have a Google Maps link already stored
    if (locationInput.value && (
        locationInput.value.includes('google.com/maps') || 
        locationInput.value.includes('maps.app.goo.gl') || 
        locationInput.value.includes('goo.gl/maps'))) {
        
        console.log("Found Google Maps link:", locationInput.value);
        // Extract coordinates
        const coords = extractCoordinatesFromMapsLink(locationInput.value);
        if (coords && coords !== true && coords.lat && coords.lng) {
            console.log("Extracted coordinates:", coords);
            // Store as data attributes
            locationInput.dataset.lat = coords.lat.toFixed(6);
            locationInput.dataset.lng = coords.lng.toFixed(6);
        }
    }
    // If we have direct coordinates instead of a Google Maps link
    else if (locationInput.value && locationInput.value.includes(',') && !locationInput.value.includes('http')) {
        console.log("Found coordinates string:", locationInput.value);
        const parts = locationInput.value.split(',');
        if (parts.length === 2) {
            const lat = parseFloat(parts[0].trim());
            const lng = parseFloat(parts[1].trim());
            
            if (!isNaN(lat) && !isNaN(lng)) {
                console.log("Parsed coordinates:", lat, lng);
                // Convert to Google Maps link
                const mapsLink = createGoogleMapsLink(lat, lng);
                locationInput.value = mapsLink;
                console.log("Created Google Maps link:", mapsLink);
                
                // Store coordinates as data attributes
                locationInput.dataset.lat = lat.toFixed(6);
                locationInput.dataset.lng = lng.toFixed(6);
                
                // Save the Google Maps link
                saveSettings({ 'broadcast.location': mapsLink });
            }
        }
    }
}

// Function to validate and parse coordinates or maps link
function validateLocationInput(input) {
    if (!input) return null;
    
    // Check if it's direct coordinates (lat,lng)
    if (input.includes(',') && !input.includes('http')) {
        try {
            const parts = input.split(',');
            if (parts.length === 2) {
                const lat = parseFloat(parts[0].trim());
                const lng = parseFloat(parts[1].trim());
                
                if (!isNaN(lat) && !isNaN(lng) && 
                    lat >= -90 && lat <= 90 && 
                    lng >= -180 && lng <= 180) {
                    return { lat, lng };
                }
            }
        } catch (e) {
            console.error("Error parsing coordinates:", e);
        }
    }
    
    // Check if it's a Google Maps link
    if (input.includes('google.com/maps') || 
        input.includes('maps.app.goo.gl') || 
        input.includes('goo.gl/maps')) {
        
        const result = extractCoordinatesFromMapsLink(input);
        if (result) {
            return result === true ? { isValid: true } : result;
        }
    }
    
    return null;
}

// Format distance in km with proper units
function formatDistance(distanceKm) {
    if (distanceKm < 1) {
        // Convert to meters for distances less than 1km
        const meters = Math.round(distanceKm * 1000);
        return `${meters} meters`;
    } else if (distanceKm < 10) {
        // Show one decimal place for small distances
        return `${distanceKm.toFixed(1)} km`;
    } else {
        // Round to nearest km for larger distances
        return `${Math.round(distanceKm)} km`;
    }
}

// Add event listener specifically for the broadcast_location field
function setupBroadcastLocationSaving() {
    const locationInput = document.getElementById('broadcast_location');
    if (locationInput) {
        console.log("Setting up broadcast location saving for:", locationInput.value);
        
        // Add blur event listener to save when the user finishes editing
        locationInput.addEventListener('blur', function() {
            console.log("Saving broadcast location on blur:", locationInput.value);
            
            // Validate input and convert to proper format if needed
            const coordinates = validateLocationInput(locationInput.value);
            if (coordinates) {
                if (coordinates.lat && coordinates.lng) {
                    // If direct coordinates were entered, convert to Google Maps link
                    if (!locationInput.value.includes('google.com/maps') && 
                        !locationInput.value.includes('goo.gl/maps') && 
                        !locationInput.value.includes('maps.app.goo.gl')) {
                        
                        const mapsLink = createGoogleMapsLink(coordinates.lat, coordinates.lng);
                        locationInput.value = mapsLink;
                    }
                    
                    // Store coordinates as data attributes
                    locationInput.dataset.lat = coordinates.lat.toFixed(6);
                    locationInput.dataset.lng = coordinates.lng.toFixed(6);
                }
                
                // Save the validated location - ensure it is saved
                saveSettingsNow({ 'broadcast.location': locationInput.value });
            } else {
                // Show error for invalid input
                showError("Please enter a valid Google Maps link or coordinates in format 'latitude,longitude'");
            }
        });
        
        // Also add input event listener with debounce
        locationInput.addEventListener('input', debounce(function() {
            console.log("Input detected in broadcast location:", locationInput.value);
            // Track that we have changes pending
            pendingChanges['broadcast.location'] = locationInput.value;
        }, 1000));
    }
}

// Immediate, non-debounced save function for critical saves
function saveSettingsNow(settings) {
    showSaving();
    
    // Debug the settings being saved
    console.log("Saving settings immediately:", settings);
    
    // Make API call to save settings
    return fetch('/update_notification_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(settings)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showSaved();
            console.log("Settings saved successfully");
            return true;
        } else {
            showError(data.error || 'Unknown error');
            console.error("Error saving settings:", data.error);
            return false;
        }
    })
    .catch(error => {
        console.error('Error saving settings:', error);
        showError('Failed to save settings');
        return false;
    });
}

// Auto-save functionality with debounce
const saveSettings = debounce(function(settings) {
    // Add settings to pending changes
    Object.assign(pendingChanges, settings);
    
    // If already saving, just add to pending changes
    if (isSaving) {
        console.log("Already saving, added to pending changes:", settings);
        return;
    }
    
    showSaving();
    
    // Debug the settings being saved
    console.log("Saving settings:", pendingChanges);
    
    // Make API call to save settings
    fetch('/update_notification_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(pendingChanges)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showSaved();
            console.log("Settings saved successfully");
        } else {
            showError(data.error || 'Unknown error');
            console.error("Error saving settings:", data.error);
        }
    })
    .catch(error => {
        console.error('Error saving settings:', error);
        showError('Failed to save settings');
    });
}, 500); // Keep original 500ms debounce time

// Initialize the page when DOM is loaded
function initializeNotificationSettings() {
    console.log("Initializing notification settings");
    
    // Add page unload handler to save pending changes
    window.addEventListener('beforeunload', function(e) {
        if (Object.keys(pendingChanges).length > 0) {
            // Save all pending changes
            saveSettingsNow(pendingChanges);
            
            // This won't block navigation but gives browser a chance to save data
            // and shows standard "unsaved changes" dialog
            e.preventDefault();
            e.returnValue = '';
        }
    });
    
    // Add page visibility change handler to save when tab is switched
    document.addEventListener('visibilitychange', function() {
        if (document.visibilityState === 'hidden' && Object.keys(pendingChanges).length > 0) {
            console.log("Page visibility changed to hidden, saving all pending changes");
            saveSettingsNow(pendingChanges);
        }
    });
    
    // Initialize intl-tel-input for SMS
    const smsInput = document.querySelector("#sms_recipient");
    let smsIti = null;
    
    if (smsInput && window.intlTelInput) {
        smsIti = window.intlTelInput(smsInput, {
            initialCountry: "in", // Default to India
            preferredCountries: ["in", "us", "gb", "ca", "au"],
            separateDialCode: true,
            utilsScript: "https://cdnjs.cloudflare.com/ajax/libs/intl-tel-input/17.0.13/js/utils.js",
        });
    }

    // Initialize intl-tel-input for Call
    const callInput = document.querySelector("#call_recipient");
    let callIti = null;
    
    if (callInput && window.intlTelInput) {
        callIti = window.intlTelInput(callInput, {
            initialCountry: "in", // Default to India
            preferredCountries: ["in", "us", "gb", "ca", "au"],
            separateDialCode: true,
            utilsScript: "https://cdnjs.cloudflare.com/ajax/libs/intl-tel-input/17.0.13/js/utils.js",
        });
    }

    // Set initial values if preferences exist
    const countryCodeField = document.getElementById('country_code');
    if (countryCodeField && countryCodeField.value && smsIti) {
        try {
            // Try to initialize with saved country code
            let countryCode = countryCodeField.value;
            if (countryCode) {
                // Remove the '+' if present
                if (countryCode.startsWith('+')) {
                    countryCode = countryCode.substring(1);
                }
                smsIti.setCountry(getCountryCodeByDialCode(countryCode));
            }
        } catch (e) {
            console.log("Error setting initial country for SMS", e);
        }
    }

    const countryCodeCallField = document.getElementById('country_code_call');
    if (countryCodeCallField && countryCodeCallField.value && callIti) {
        try {
            // Try to initialize with saved country code
            let countryCodeCall = countryCodeCallField.value;
            if (countryCodeCall) {
                // Remove the '+' if present
                if (countryCodeCall.startsWith('+')) {
                    countryCodeCall = countryCodeCall.substring(1);
                }
                callIti.setCountry(getCountryCodeByDialCode(countryCodeCall));
            }
        } catch (e) {
            console.log("Error setting initial country for Call", e);
        }
    }

    // Save country codes when phone number changes
    if (smsInput && smsIti) {
        smsInput.addEventListener('change', function() {
            const countryCodeField = document.getElementById('country_code');
            if (countryCodeField && smsIti) {
                countryCodeField.value = '+' + smsIti.getSelectedCountryData().dialCode;
                saveSettings({ 'sms.country_code': '+' + smsIti.getSelectedCountryData().dialCode });
            }
        });
        
        // Additionally save on blur
        smsInput.addEventListener('blur', function() {
            const countryCodeField = document.getElementById('country_code');
            if (countryCodeField && smsIti) {
                countryCodeField.value = '+' + smsIti.getSelectedCountryData().dialCode;
                const settings = {
                    'sms.country_code': '+' + smsIti.getSelectedCountryData().dialCode,
                    'sms.recipient': smsInput.value
                };
                saveSettings(settings);
            }
        });
    }
    
    if (callInput && callIti) {
        callInput.addEventListener('change', function() {
            const countryCodeCallField = document.getElementById('country_code_call');
            if (countryCodeCallField && callIti) {
                countryCodeCallField.value = '+' + callIti.getSelectedCountryData().dialCode;
                saveSettings({ 'call.country_code': '+' + callIti.getSelectedCountryData().dialCode });
            }
        });
        
        // Additionally save on blur
        callInput.addEventListener('blur', function() {
            const countryCodeCallField = document.getElementById('country_code_call');
            if (countryCodeCallField && callIti) {
                countryCodeCallField.value = '+' + callIti.getSelectedCountryData().dialCode;
                const settings = {
                    'call.country_code': '+' + callIti.getSelectedCountryData().dialCode,
                    'call.recipient': callInput.value
                };
                saveSettings(settings);
            }
        });
    }

    // Copy from Voice Call button
    const copyFromVoiceCallBtn = document.getElementById('copy_from_voice_call');
    if (copyFromVoiceCallBtn && smsIti && callIti && smsInput && callInput) {
        copyFromVoiceCallBtn.addEventListener('click', function () {
            smsInput.value = callInput.value;
            smsIti.setCountry(callIti.getSelectedCountryData().iso2);
            
            // Save the updated SMS number
            saveSettings({ 
                'sms.recipient': callInput.value,
                'sms.country_code': '+' + callIti.getSelectedCountryData().dialCode
            });
            
            // Show notification
            Swal.fire({
                title: 'Success',
                text: 'Phone number copied from Voice Call settings',
                icon: 'success',
                timer: 2000,
                showConfirmButton: false
            });
        });
    }

    // Toggle sections opacity based on checkboxes
    const emailEnabledCheckbox = document.getElementById('email_enabled');
    const emailSettings = document.querySelector('.email-settings');

    if (emailEnabledCheckbox && emailSettings) {
        emailEnabledCheckbox.addEventListener('change', function () {
            if (this.checked) {
                emailSettings.classList.remove('opacity-50');
            } else {
                emailSettings.classList.add('opacity-50');
            }
        });
    }

    const smsEnabledCheckbox = document.getElementById('sms_enabled');
    const smsSettings = document.querySelector('.sms-settings');

    if (smsEnabledCheckbox && smsSettings) {
        smsEnabledCheckbox.addEventListener('change', function () {
            if (this.checked) {
                smsSettings.classList.remove('opacity-50');
            } else {
                smsSettings.classList.add('opacity-50');
            }
        });
    }

    const telegramEnabledCheckbox = document.getElementById('telegram_enabled');
    const telegramSettings = document.querySelector('.telegram-settings');

    if (telegramEnabledCheckbox && telegramSettings) {
        telegramEnabledCheckbox.addEventListener('change', function () {
            if (this.checked) {
                telegramSettings.classList.remove('opacity-50');
            } else {
                telegramSettings.classList.add('opacity-50');
            }
        });
    }

    const callEnabledCheckbox = document.getElementById('call_enabled');
    const callSettings = document.querySelector('.call-settings');

    if (callEnabledCheckbox && callSettings) {
        callEnabledCheckbox.addEventListener('change', function () {
            if (this.checked) {
                callSettings.classList.remove('opacity-50');
            } else {
                callSettings.classList.add('opacity-50');
            }
        });
    }

    // Broadcast toggle
    const broadcastEnabledCheckbox = document.getElementById('broadcast_enabled');
    const broadcastSettings = document.querySelector('.broadcast-settings');

    if (broadcastEnabledCheckbox && broadcastSettings) {
        broadcastEnabledCheckbox.addEventListener('change', function () {
            if (this.checked) {
                broadcastSettings.classList.remove('opacity-50');
            } else {
                broadcastSettings.classList.add('opacity-50');
            }
        });
    }

    // Update radius value display
    const broadcastRadiusInput = document.getElementById('broadcast_radius');
    const radiusValueDisplay = document.getElementById('radius_value');

    if (broadcastRadiusInput && radiusValueDisplay) {
        broadcastRadiusInput.addEventListener('input', function() {
            radiusValueDisplay.textContent = this.value;
            // Track changes
            pendingChanges['broadcast.radius'] = parseInt(this.value);
        });
        
        // Save when user releases the slider
        broadcastRadiusInput.addEventListener('change', function() {
            radiusValueDisplay.textContent = this.value;
            saveSettings({ 'broadcast.radius': parseInt(this.value) });
        });
    }

    // "All animals" checkbox logic
    const animalAllCheckbox = document.getElementById('animal_all');
    const animalCheckboxes = document.querySelectorAll('input[id^="animal_"]:not([id="animal_all"])');
    
    if (animalAllCheckbox) {
        animalAllCheckbox.addEventListener('change', updateAnimalCheckboxes);

        // Individual animal checkbox logic - uncheck "all" if any individual is unchecked
        animalCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                // If this checkbox is unchecked, uncheck the "all" checkbox
                if (!this.checked && animalAllCheckbox.checked) {
                    animalAllCheckbox.checked = false;
                    
                    // Enable all individual checkboxes when "All" is unchecked
                    animalCheckboxes.forEach(cb => {
                        cb.disabled = false;
                    });
                }
                
                // If all individual checkboxes are checked, check the "all" checkbox
                if (Array.from(animalCheckboxes).every(cb => cb.checked)) {
                    animalAllCheckbox.checked = true;
                    
                    // Disable individual checkboxes when "All" becomes checked
                    animalCheckboxes.forEach(cb => {
                        cb.disabled = true;
                    });
                }
                
                // Save animal selections
                saveAnimalSelections();
            });
        });

        // Check if "All animals" is already checked and set initial state
        if (animalAllCheckbox.checked) {
            animalCheckboxes.forEach(checkbox => {
                checkbox.checked = true;
                checkbox.disabled = true;
            });
        }
    }
    
    // Add auto-save listeners to all inputs with auto-save class
    document.querySelectorAll('.auto-save').forEach(input => {
        input.addEventListener('change', function() {
            const setting = this.getAttribute('data-setting');
            const value = this.type === 'checkbox' ? this.checked : this.value;
            
            const settings = {};
            settings[setting] = value;
            console.log(`Auto-saving ${setting}:`, value);
            saveSettings(settings);
        });
        
        // For text inputs, also listen for blur event (for when user finishes typing)
        if (input.type === 'text' || input.type === 'email' || input.type === 'tel') {
            // Track input changes
            input.addEventListener('input', function() {
                const setting = this.getAttribute('data-setting');
                pendingChanges[setting] = this.value;
            });
            
            input.addEventListener('blur', function() {
                const setting = this.getAttribute('data-setting');
                const value = this.value;
                
                const settings = {};
                settings[setting] = value;
                console.log(`Auto-saving ${setting} on blur:`, value);
                saveSettings(settings);
            });
        }
    });
    
    // Auto-save for animal checkboxes
    document.querySelectorAll('.auto-save-animal').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            saveAnimalSelections();
        });
    });
    
    // Auto-save for notification method checkboxes
    document.querySelectorAll('.auto-save-method').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            saveNotificationMethods();
        });
    });
    
    // Initialize location input and handle Google Maps links
    setupLocationInput();
    
    // Add specific handler for broadcast location
    setupBroadcastLocationSaving();
    
    // Initial setup for "All animals" checkbox
    if (animalAllCheckbox) {
        updateAnimalCheckboxes();
    }
    
    // Remove the "Pick on Map" button
    const showMapBtn = document.getElementById('show_map');
    if (showMapBtn) {
        showMapBtn.style.display = 'none';
    }
    
    // Hide the map container permanently
    const mapContainer = document.getElementById('map-container');
    if (mapContainer) {
        mapContainer.style.display = 'none';
        mapContainer.classList.add('hidden');
    }
    
    // Hide the search button if it exists
    const searchLocationBtn = document.getElementById('search_location');
    if (searchLocationBtn) {
        searchLocationBtn.style.display = 'none';
    }
    
    console.log("Notification settings initialization complete");
}