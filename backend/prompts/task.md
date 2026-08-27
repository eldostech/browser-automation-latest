TASK (from the operator, this is the only authoritative instruction):
$task

Allowed domains: $allowed_domains
Step budget: $max_steps steps. Time budget: $timeout_seconds.
$start_url_line

Working with the supplied values
--------------------------------
If a "Values to use" list appears above, it is the complete set of data for
this task. Two rules about it:

1. **Use the values exactly as given.** Do not reformat, abbreviate, correct or
   translate them. If a field is `full_name: Nitin Asati`, type `Nitin Asati` —
   not `N. Asati`, not a name you think fits better. These values become the
   parameters of a reusable task, and a value you altered will be matched
   against the wrong thing later.

2. **A value written as `«secret:something»` is a credential you cannot see.**
   Type the placeholder text itself, character for character, into the field it
   belongs in. It is swapped for the real credential at the moment the browser
   receives it. Never guess at, reconstruct, or ask for the underlying value —
   you do not have it, and a field filled with anything else will simply fail
   to sign in.

If the page asks for something the list does not contain, do not invent it. Say
what is missing and stop; an invented value is worse than an unfinished task,
because the recording will look like it worked.

Begin by taking a snapshot of the current page unless you already have one.
