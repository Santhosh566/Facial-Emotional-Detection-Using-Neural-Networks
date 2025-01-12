<!DOCTYPE html>
<html>
<head>
<style>
    /* Style for the links */
    .link {
        color: blue;
        text-decoration: underline;
        cursor: pointer;
    }
    
    /* Style for the links when hovered */
    .link:hover {
        color: red;
        text-decoration: none; /* Remove underline on hover */
    }
</style>
</head>
<body>
    <!-- First link with onclick event -->
    <p><span class="link" onclick="openLink('https://www.example.com/link1')">Link 1</span></p>
    
    <!-- Second link with onclick event -->
    <p><span class="link" onclick="openLink('https://www.example.com/link2')">Link 2</span></p>

    <script>
        // JavaScript function to open links
        function openLink(url) {
            // Open the link in a new tab
            window.open(url, '_blank');
            // Perform another action here if needed
            // For example, display an alert
            alert('Link opened: ' + url);
        }
    </script>
</body>
</html>
