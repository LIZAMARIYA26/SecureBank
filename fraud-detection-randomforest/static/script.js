(function () {
  const root = document.body;
  const toggle = document.getElementById("themeToggle");
  const spinner = document.getElementById("spinner");

  const saved = localStorage.getItem("securebank-theme");
  if (saved === "dark") {
    root.classList.add("dark");
    if (toggle) toggle.textContent = "Light mode";
  }

  if (toggle) {
    toggle.addEventListener("click", function () {
      const dark = root.classList.toggle("dark");
      localStorage.setItem("securebank-theme", dark ? "dark" : "light");
      toggle.textContent = dark ? "Light mode" : "Dark mode";
    });
  }

  document.querySelectorAll("form[data-loading]").forEach(function (form) {
    form.addEventListener("submit", function (event) {
      if (form.hasAttribute("data-validate")) {
        const amount = form.querySelector("[name=amount]");
        const hour = form.querySelector("[name=hour]");
        const freq = form.querySelector("[name=tx_frequency]");
        const age = form.querySelector("[name=age]");
        const messages = [];
        if (amount && Number(amount.value) <= 0) messages.push("Amount must be greater than 0.");
        if (hour && (Number(hour.value) < 0 || Number(hour.value) > 23)) messages.push("Hour must be 0–23.");
        if (freq && Number(freq.value) < 1) messages.push("Frequency must be at least 1.");
        if (age && (Number(age.value) < 16 || Number(age.value) > 90)) messages.push("Age must be between 16 and 90.");
        if (messages.length) {
          event.preventDefault();
          alert(messages.join("\n"));
          return;
        }
      }
      if (spinner) spinner.classList.remove("hidden");
    });
  });


})();
