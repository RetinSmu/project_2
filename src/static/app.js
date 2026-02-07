const q = document.getElementById("q");
const run = document.getElementById("run");
const status = document.getElementById("status");
const vanilla = document.getElementById("vanilla");
const agentic = document.getElementById("agentic");

function setStatus(msg, kind = "") {
  status.className = "status " + kind;
  status.textContent = msg;
}

async function ask() {
  const question = q.value.trim();
  if (!question) return;

  setStatus("Running...");
  run.disabled = true;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();

    if (!res.ok) {
      setStatus(data.error || "Server error", "err");
      return;
    }

    vanilla.innerHTML = data.vanilla_html || "<p>No answer.</p>";
    agentic.innerHTML = data.agentic_html || "<p>No answer.</p>";
    setStatus("");

  } catch (e) {
    setStatus(String(e), "err");
  } finally {
    run.disabled = false;
  }
}

run.addEventListener("click", ask);
q.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});

document.querySelectorAll(".chip").forEach((c) => {
  c.addEventListener("click", () => {
    q.value = c.textContent;
    ask();
  });
});
