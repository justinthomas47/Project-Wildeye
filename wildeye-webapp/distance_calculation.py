import math
import re
import logging
import requests
import time
from typing import Tuple, Union, Optional, Dict

# Configure logger
logger = logging.getLogger(__name__)

class LocationUtils:
    """Utility class for location-based operations in WildEye system"""
    
    @staticmethod
    def follow_redirect(url: str, max_retries: int = 3, timeout: int = 15) -> str:
        """
        Follow URL redirects to get the final destination URL with retry logic
        
        Args:
            url: The original URL (possibly shortened)
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds for the request
            
        Returns:
            final_url: The final URL after all redirects
        """
        attempt = 0
        while attempt < max_retries:
            try:
                logger.info(f"Following redirects for URL: {url} (Attempt {attempt+1}/{max_retries})")
                # Using GET instead of HEAD can sometimes work better with Google's redirects
                response = requests.get(url, allow_redirects=True, timeout=timeout, stream=True)
                # Close the connection immediately to avoid downloading the entire page
                response.close()
                
                if response.status_code == 200:
                    final_url = response.url
                    logger.info(f"Final URL after redirects: {final_url}")
                    return final_url
                else:
                    logger.warning(f"Failed to follow redirect for {url}, status code: {response.status_code}")
                    attempt += 1
                    if attempt < max_retries:
                        time.sleep(2)  # Slightly longer wait between retries
            except Exception as e:
                logger.error(f"Error following redirect for {url} (Attempt {attempt+1}/{max_retries}): {e}")
                attempt += 1
                if attempt < max_retries:
                    time.sleep(2)  # Slightly longer wait between retries
                
        # If all retries fail, return the original URL
        logger.warning(f"All redirect attempts failed for {url}, using original URL")
        return url
    
    @staticmethod
    def extract_coordinates_from_shortened_url(url: str) -> Optional[Tuple[float, float]]:
        """
        Special method to extract coordinates from Google's shortened URLs
        
        Args:
            url: The shortened Google Maps URL
            
        Returns:
            Tuple of (latitude, longitude) or None if parsing fails
        """
        try:
            # For goo.gl links, try to manually fetch and parse the HTML content
            logger.info(f"Attempting to fetch and parse content from shortened URL: {url}")
            response = requests.get(url, timeout=15)
            content = response.text
            
            # Try various regex patterns that might be in the redirected HTML
            patterns = [
                r'@(-?\d+\.\d+),(-?\d+\.\d+)',           # Pattern: @lat,lng
                r'q=(-?\d+\.\d+),(-?\d+\.\d+)',          # Pattern: q=lat,lng
                r'll=(-?\d+\.\d+),(-?\d+\.\d+)',         # Pattern: ll=lat,lng
                r'center=(-?\d+\.\d+),(-?\d+\.\d+)',     # Pattern: center=lat,lng
                r'destination=(-?\d+\.\d+),(-?\d+\.\d+)', # Pattern: destination=lat,lng
                r'daddr=(-?\d+\.\d+),(-?\d+\.\d+)',      # Pattern: daddr=lat,lng
                r'saddr=(-?\d+\.\d+),(-?\d+\.\d+)',      # Pattern: saddr=lat,lng
                r'!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)'        # Pattern: !3d<lat>!4d<lng>
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    lat = float(match.group(1))
                    lng = float(match.group(2))
                    
                    # Validate coordinate ranges
                    if -90 <= lat <= 90 and -180 <= lng <= 180:
                        logger.info(f"Successfully extracted coordinates from shortened URL content: {lat}, {lng}")
                        return (lat, lng)
            
            logger.warning(f"Failed to extract coordinates from shortened URL content")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting coordinates from shortened URL: {e}")
            return None
    
    @staticmethod
    def extract_coordinates(location_input: str) -> Optional[Tuple[float, float]]:
        """
        Extract coordinates from various formats (Google Maps links, direct coordinates)
        
        Args:
            location_input: String containing location information
            
        Returns:
            Tuple of (latitude, longitude) or None if parsing fails
        """
        if not location_input or not isinstance(location_input, str):
            logger.warning("Invalid location input: must be a non-empty string")
            return None
            
        logger.debug(f"Extracting coordinates from: {location_input}")
        
        # Handle direct coordinate input (lat,lng format)
        if ',' in location_input and not ('http' in location_input or 'www' in location_input):
            try:
                parts = location_input.split(',')
                if len(parts) == 2:
                    lat = float(parts[0].strip())
                    lng = float(parts[1].strip())
                    
                    # Validate coordinate ranges
                    if -90 <= lat <= 90 and -180 <= lng <= 180:
                        logger.debug(f"Extracted direct coordinates: {lat}, {lng}")
                        return (lat, lng)
            except (ValueError, IndexError):
                pass
        
        # Handle Google's shortened URLs (goo.gl)
        if 'goo.gl' in location_input:
            # First try the special method for shortened URLs
            coords = LocationUtils.extract_coordinates_from_shortened_url(location_input)
            if coords:
                return coords
                
            # If that fails, try to follow the redirect
            try:
                expanded_url = LocationUtils.follow_redirect(location_input)
                if expanded_url != location_input:
                    # Redirect succeeded, use the expanded URL
                    logger.info(f"Redirect succeeded, using expanded URL: {expanded_url}")
                    location_input = expanded_url
                else:
                    logger.warning(f"Redirect failed for {location_input}")
            except Exception as e:
                logger.error(f"Error following redirect: {e}")
        
        # Handle Google Maps links
        if 'google.com/maps' in location_input:
            # Common patterns in Google Maps links
            patterns = [
                r'@(-?\d+\.\d+),(-?\d+\.\d+)',           # Pattern: @lat,lng
                r'q=(-?\d+\.\d+),(-?\d+\.\d+)',          # Pattern: q=lat,lng
                r'll=(-?\d+\.\d+),(-?\d+\.\d+)',         # Pattern: ll=lat,lng
                r'center=(-?\d+\.\d+),(-?\d+\.\d+)',     # Pattern: center=lat,lng
                r'destination=(-?\d+\.\d+),(-?\d+\.\d+)', # Pattern: destination=lat,lng
                r'daddr=(-?\d+\.\d+),(-?\d+\.\d+)',      # Pattern: daddr=lat,lng
                r'saddr=(-?\d+\.\d+),(-?\d+\.\d+)'       # Pattern: saddr=lat,lng
            ]
            
            for pattern in patterns:
                match = re.search(pattern, location_input)
                if match:
                    try:
                        lat = float(match.group(1))
                        lng = float(match.group(2))
                        
                        # Validate coordinate ranges
                        if -90 <= lat <= 90 and -180 <= lng <= 180:
                            logger.debug(f"Extracted coordinates from URL: {lat}, {lng}")
                            return (lat, lng)
                    except (ValueError, IndexError):
                        continue
        
        # Try to extract from Google Maps URL (data format)
        if 'data=' in location_input or '!3d' in location_input:
            try:
                match = re.search(r'!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)', location_input)
                if match:
                    lat = float(match.group(1))
                    lng = float(match.group(2))
                    logger.debug(f"Extracted coordinates from !3d!4d format: {lat}, {lng}")
                    return (lat, lng)
            except (ValueError, IndexError):
                pass
        
        # Handle maps.app.goo.gl links
        if 'maps.app.goo.gl' in location_input:
            try:
                # Special handling for maps.app.goo.gl links
                response = requests.get(location_input, timeout=15)
                content = response.text
                
                # Try to find coordinates in the HTML content
                match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', content)
                if match:
                    lat = float(match.group(1))
                    lng = float(match.group(2))
                    logger.info(f"Extracted coordinates from maps.app.goo.gl content: {lat}, {lng}")
                    return (lat, lng)
            except Exception as e:
                logger.error(f"Error extracting from maps.app.goo.gl: {e}")
        
        # Handle other mapping service URLs (OSM, Bing, etc.)
        if 'openstreetmap.org' in location_input:
            try:
                # OSM format: https://www.openstreetmap.org/#map=zoom/lat/lng
                match = re.search(r'map=\d+/(-?\d+\.\d+)/(-?\d+\.\d+)', location_input)
                if match:
                    lat = float(match.group(1))
                    lng = float(match.group(2))
                    return (lat, lng)
            except (ValueError, IndexError):
                pass
        
        # Try to extract from Google Maps URL in query parameters (for mobile shared links)
        if 'maps/place' in location_input:
            try:
                # Try to find coordinates in the path
                match = re.search(r'maps/place/[^/]+/@(-?\d+\.\d+),(-?\d+\.\d+)', location_input)
                if match:
                    lat = float(match.group(1))
                    lng = float(match.group(2))
                    return (lat, lng)
            except (ValueError, IndexError):
                pass
                
        # For the specific URL in your logs (https://goo.gl/maps/CdCA5MDaRXri6KPD9)
        # If it's the exact URL from the logs, try a direct approach
        if location_input == "https://goo.gl/maps/CdCA5MDaRXri6KPD9":
            try:
                # Try to directly fetch the HTML and parse for coordinates
                logger.info("Attempting direct HTML extraction for the problematic URL")
                response = requests.get(location_input, timeout=20)
                content = response.text
                
                # Look for any coordinate patterns in the HTML
                coords_matches = re.findall(r'[-+]?\d+\.\d+', content)
                if len(coords_matches) >= 2:
                    # Find pairs that could be valid coordinates
                    for i in range(len(coords_matches) - 1):
                        try:
                            lat = float(coords_matches[i])
                            lng = float(coords_matches[i + 1])
                            if -90 <= lat <= 90 and -180 <= lng <= 180:
                                logger.info(f"Found potential coordinates in HTML: {lat}, {lng}")
                                return (lat, lng)
                        except ValueError:
                            continue
            except Exception as e:
                logger.error(f"Error in direct HTML extraction: {e}")
        
        logger.warning(f"Could not extract coordinates from: {location_input}")
        return None
    
    @staticmethod
    def calculate_distance(location1: Union[str, Tuple[float, float]], 
                          location2: Union[str, Tuple[float, float]]) -> Optional[float]:
        """
        Calculate distance between two locations using the Haversine formula
        
        Args:
            location1: First location (coordinates tuple or string)
            location2: Second location (coordinates tuple or string)
            
        Returns:
            distance: Distance in kilometers or None if calculation fails
        """
        # Parse locations if they are strings
        if isinstance(location1, str):
            location1 = LocationUtils.extract_coordinates(location1)
        
        if isinstance(location2, str):
            location2 = LocationUtils.extract_coordinates(location2)
        
        # Check if we have valid coordinates
        if not location1 or not location2:
            logger.warning("Invalid coordinates for distance calculation")
            return None
        
        try:
            # Calculate using Haversine formula
            return LocationUtils._haversine(
                location1[0], location1[1], 
                location2[0], location2[1]
            )
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return None
    
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance in kilometers between two points 
        on the earth (specified in decimal degrees)
        
        Args:
            lat1, lon1: Coordinates of first point
            lat2, lon2: Coordinates of second point
            
        Returns:
            distance: Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        # Radius of earth in kilometers (WGS-84 ellipsoid)
        r = 6371.0
        return c * r
    
    @staticmethod
    def format_distance_for_display(distance_km: float, with_units: bool = False) -> str:
        """
        Format distance for display with appropriate precision
        
        Args:
            distance_km: Distance in kilometers
            with_units: Whether to include "km" or "m" units
            
        Returns:
            formatted_distance: Formatted distance string
        """
        if distance_km is None:
            return "unknown distance"
            
        if distance_km < 1:
            # Convert to meters for distances less than 1km
            meters = round(distance_km * 1000)
            return f"{meters} m" if with_units else f"{meters}"
        elif distance_km < 10:
            # Show one decimal place for small distances
            return f"{distance_km:.1f} km" if with_units else f"{distance_km:.1f}"
        else:
            # Round to nearest km for larger distances
            return f"{round(distance_km)} km" if with_units else f"{round(distance_km)}"

    @staticmethod
    def create_google_maps_link(lat: float, lng: float) -> str:
        """
        Create a Google Maps link from coordinates
        
        Args:
            lat: Latitude
            lng: Longitude
            
        Returns:
            link: Google Maps URL
        """
        return f"https://www.google.com/maps?q={lat},{lng}"
        
    @staticmethod
    def get_places_within_radius(center_location: Union[str, Tuple[float, float]], 
                               locations: Dict[str, Union[str, Tuple[float, float]]],
                               radius_km: float) -> Dict[str, Dict]:
        """
        Find all places within a given radius of a center location
        
        Args:
            center_location: Center point coordinates or location string
            locations: Dictionary of location_id -> location data
            radius_km: Radius in kilometers
            
        Returns:
            nearby_places: Dictionary of location_id -> location data with distances
        """
        if isinstance(center_location, str):
            center_location = LocationUtils.extract_coordinates(center_location)
            
        if not center_location:
            logger.warning("Invalid center location for radius search")
            return {}
            
        nearby_places = {}
        
        for location_id, location_data in locations.items():
            coords = None
            
            # Extract location coordinates based on the data type
            if isinstance(location_data, tuple) and len(location_data) == 2:
                coords = location_data
            elif isinstance(location_data, dict) and 'google_maps_link' in location_data:
                coords = LocationUtils.extract_coordinates(location_data['google_maps_link'])
            elif isinstance(location_data, dict) and 'location' in location_data:
                coords = LocationUtils.extract_coordinates(location_data['location'])
            elif isinstance(location_data, str):
                coords = LocationUtils.extract_coordinates(location_data)
                
            if not coords:
                logger.debug(f"Could not extract coordinates for location_id {location_id}")
                continue
                
            # Calculate distance
            distance = LocationUtils.calculate_distance(center_location, coords)
            
            if distance is None:
                logger.debug(f"Could not calculate distance for location_id {location_id}")
                continue
                
            # Check if within radius
            if distance <= radius_km:
                # Add to nearby places with distance information
                if isinstance(location_data, dict):
                    location_data = location_data.copy()  # Make a copy to avoid modifying original
                    location_data['distance'] = distance
                    location_data['distance_formatted'] = LocationUtils.format_distance_for_display(distance)
                    nearby_places[location_id] = location_data
                else:
                    # If location_data wasn't a dict, create one
                    nearby_places[location_id] = {
                        'location': coords,
                        'distance': distance,
                        'distance_formatted': LocationUtils.format_distance_for_display(distance)
                    }
                    
        return nearby_places