document.addEventListener('DOMContentLoaded', function() {
    console.log("Dashboard Loaded");

    // Example of a custom action on dashboard load, you can replace with more complex logic
    setTimeout(function() {
        document.getElementById("total-transactions").innerText = "120,000";
        document.getElementById("fraud-cases").innerText = "1,200";
        document.getElementById("fraud-percentage").innerText = "1.0%";
    }, 2000);
});
